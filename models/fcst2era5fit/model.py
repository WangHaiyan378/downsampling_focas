"""
Task B: Forecast2ERA5Fit Model
Input: Multi-model future forecasts [B, M, K, C_in, H, W]
       Static features [B, C_static, H, W]
       Lead IDs [B, K]
Output: Bias-corrected ERA5 uv + convergence

Architecture:
1. Per-model forecast encoder
2. Lead-aware model weighting
3. Terrain / LCZ conditioning
4. Spatial residual correction backbone (UNet-like)
5. Residual output: corrected = weighted_mean + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.conv_blocks import ResBlock, DoubleConv, DownBlock, UpBlock
from models.common.attention import SpatialAttention, MultiHeadSelfAttention
from models.common.conditioning import TerrainLCZConditioner
from models.common.lead_embed import LeadTimeEmbedding, SpatialLeadModulation
from models.common.output_head import UVConvOutputHead

from models.fcst2era5fit.wavelet_lcz_encoder import WaveletLCZForecastEncoder


class ForecastEncoder(nn.Module):
    """Encode one model's forecast at K lead times: [B, K, C_in, H, W] -> [B, D, H, W]"""

    def __init__(self, c_in, d_model, num_leads):
        super().__init__()
        self.per_lead_conv = nn.Sequential(
            DoubleConv(c_in, d_model // 2),
            ResBlock(d_model // 2),
            nn.Conv2d(d_model // 2, d_model, 1),
        )
        self.lead_aggregator = nn.Sequential(
            nn.Conv2d(d_model * num_leads, d_model, 1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.GELU(),
            ResBlock(d_model),
        )
        self.num_leads = num_leads

    def forward(self, x):
        """
        x: (B, K, C_in, H, W)
        returns: (B, D, H, W)
        """
        B, K, C, H, W = x.shape
        chunk_k = 8
        parts = []
        for k0 in range(0, K, chunk_k):
            k1 = min(k0 + chunk_k, K)
            xf = x[:, k0:k1].contiguous().reshape(-1, C, H, W)
            f = self.per_lead_conv(xf)
            parts.append(f.reshape(B, k1 - k0, f.shape[1], H, W))
        feats = torch.cat(parts, dim=1)
        D = feats.shape[2]
        feats = feats.reshape(B, K * D, H, W)
        return self.lead_aggregator(feats)


class LeadAwareModelWeighting(nn.Module):
    """Compute lead-dependent weights for each model."""

    def __init__(self, d_model, num_models, num_leads, n_heads=4, dropout=0.1):
        super().__init__()
        self.num_models = num_models
        self.lead_proj = nn.Linear(d_model, d_model)
        self.model_proj = nn.Linear(d_model, d_model)
        self.attn = nn.ModuleList([
            MultiHeadSelfAttention(d_model, n_heads, dropout) for _ in range(2)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, model_features, lead_emb):
        """
        model_features: (B, M, D, H, W)
        lead_emb: (B, K, D)
        returns: (B, D, H, W) weighted fused features
        """
        B, M, D, H, W = model_features.shape
        model_pool = model_features.mean(dim=(-2, -1))
        lead_mean = lead_emb.mean(dim=1, keepdim=True).expand_as(model_pool)

        combined = self.model_proj(model_pool) + self.lead_proj(lead_mean)
        for attn in self.attn:
            combined = attn(combined)

        weights = self.gate(combined)
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        fused = (model_features * weights).sum(dim=1)
        return fused


class ResidualCorrectionBackbone(nn.Module):
    """UNet-style backbone for spatial residual correction."""

    def __init__(self, d_model, n_blocks=4, dropout=0.1):
        super().__init__()
        dims = [d_model, d_model * 2, d_model * 4, d_model * 4]
        dims = dims[:n_blocks]
        while len(dims) < n_blocks:
            dims.append(dims[-1])

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.in_conv = DoubleConv(d_model, dims[0])

        for i in range(len(dims) - 1):
            self.encoder_blocks.append(DownBlock(dims[i], dims[i + 1]))

        self.bottleneck = nn.Sequential(
            ResBlock(dims[-1], dropout=dropout),
            SpatialAttention(dims[-1], n_heads=4, spatial_downsample=2, dropout=dropout),
            ResBlock(dims[-1], dropout=dropout),
        )

        for i in range(len(dims) - 1, 0, -1):
            self.decoder_blocks.append(UpBlock(dims[i], dims[i - 1], dims[i - 1]))

        self.out_conv = nn.Sequential(
            ResBlock(dims[0], dropout=dropout),
            nn.Conv2d(dims[0], d_model, 1),
        )

    def forward(self, x):
        """x: (B, D, H, W) -> (B, D, H, W) residual"""
        skips = []
        x = self.in_conv(x)
        skips.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        skips = skips[:-1]
        for i, dec in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]
            x = dec(x, skip)

        return self.out_conv(x)


class Forecast2ERA5FitModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_models = cfg["num_models"]
        self.num_leads = cfg["num_leads"]
        d = cfg["d_model"]
        c_in = cfg["c_in"]
        c_static = cfg["c_static"]
        n_heads = cfg["n_heads"]
        dropout = cfg.get("dropout", 0.1)

        enc_h = int(cfg.get("encoder_grid_h", cfg.get("grid_h", 121)))
        enc_w = int(cfg.get("encoder_grid_w", cfg.get("grid_w", 240)))
        out_h = int(cfg.get("grid_h", enc_h))
        out_w = int(cfg.get("grid_w", enc_w))
        self._enc_hw = (enc_h, enc_w)
        self._out_hw = (out_h, out_w)
        self._needs_spatial_up = (enc_h, enc_w) != (out_h, out_w)

        # 為了讓 per-model encoder 內可使用 LCZ window cross-attention + 2D RoPE，
        # 把 lead_embed 提到 encoder 之前構建，forward 時也先算 lead_emb 再餵 encoder。
        self.lead_embed = LeadTimeEmbedding(d, max_lead=120)

        self.per_model_encoders = nn.ModuleList([
            WaveletLCZForecastEncoder(
                c_in=c_in,
                c_static=c_static,
                d_model=d,
                num_leads=self.num_leads,
                wavelet_levels=cfg.get("wavelet_levels", 2),
                wavelet_include_raw=cfg.get("wavelet_include_raw", True),
                wavelet_type=cfg.get("wavelet_type", "haar"),
                use_lcz_window_cross_attention=cfg.get("use_lcz_window_cross_attention", True),
                use_lcz_2d_rope=cfg.get("use_lcz_2d_rope", True),
                lcz_attn_window_size=cfg.get("lcz_attn_window_size", 8),
                lcz_attn_heads=cfg.get("lcz_attn_heads", 4),
                lcz_attn_dropout=cfg.get("lcz_attn_dropout", dropout),
                lcz_rope_base=cfg.get("lcz_rope_base", 10000.0),
                use_lcz_scale_gate=cfg.get("use_lcz_scale_gate", True),
                scale_gate_hidden_dim=cfg.get("scale_gate_hidden_dim", d),
                scale_gate_dropout=cfg.get("scale_gate_dropout", dropout),
                dropout=dropout,
            )
            for _ in range(self.num_models)
        ])

        # 預設關閉融合後的 TerrainLCZConditioner（per-model encoder 內部已注入 LCZ）。
        # 想保留二次靜態條件修正可在 cfg 設 use_post_lcz_conditioner: true。
        self.use_post_lcz_conditioner = bool(cfg.get("use_post_lcz_conditioner", False))

        self.model_weighting = LeadAwareModelWeighting(
            d, self.num_models, self.num_leads,
            n_heads=n_heads, dropout=dropout,
        )

        self.conditioner = TerrainLCZConditioner(c_static, d)

        self.lead_spatial = SpatialLeadModulation(d)

        self.correction_backbone = ResidualCorrectionBackbone(
            d,
            n_blocks=cfg.get("n_correction_blocks", 4),
            dropout=dropout,
        )

        self.spatial_up = None
        if self._needs_spatial_up:
            self.spatial_up = nn.Conv2d(d, d, kernel_size=1)

        self.c_out_uv = int(cfg.get("c_out_uv", 2))
        self.c_out_conv = int(cfg.get("c_out_conv", 1))

        self.output_head = UVConvOutputHead(
            d, self.num_leads,
            c_out_uv=self.c_out_uv,
            c_out_conv=self.c_out_conv,
            grid_h=out_h,
            grid_w=out_w,
        )

        self.input_mean_proj = nn.Conv2d(
            c_in * self.num_leads, self.num_leads * self.c_out_uv, 1,
        )

    def forward(self, x_forecast, x_static, lead_ids):
        """
        x_forecast: (B, M, K, C_in, H, W)   -- H, W 為 S2S/encoder 網格
        x_static:   (B, C_static, H_s, W_s) -- 可能與 (H, W) 不同；模組內部會 resize
        lead_ids:   (B, K)
        """
        B, M, K, C, H, W = x_forecast.shape

        lead_emb = self.lead_embed(lead_ids)  # [B, K, D]

        model_feats = []
        for m in range(M):
            feat = self.per_model_encoders[m](
                x_forecast[:, m],
                x_static=x_static,
                lead_emb=lead_emb,
            )
            model_feats.append(feat)
        model_feats = torch.stack(model_feats, dim=1)  # [B, M, D, H, W]

        fused = self.model_weighting(model_feats, lead_emb)

        if self.use_post_lcz_conditioner:
            fused = self.conditioner(fused, x_static)

        fused = self.lead_spatial(fused, lead_emb)

        residual = self.correction_backbone(fused)

        corrected_features = fused + residual

        if self._needs_spatial_up:
            oh, ow = self._out_hw
            corrected_features = F.interpolate(
                corrected_features, size=(oh, ow), mode="bilinear", align_corners=False,
            )
            corrected_features = self.spatial_up(corrected_features)

        output = self.output_head(corrected_features)

        Cuv = self.c_out_uv
        forecast_mean = x_forecast.mean(dim=1)
        forecast_flat = forecast_mean.reshape(B, K * C, H, W)
        baseline_uv_lr = self.input_mean_proj(forecast_flat)
        baseline_uv_lr = baseline_uv_lr.reshape(B, K, Cuv, H, W)
        if self._needs_spatial_up:
            oh, ow = self._out_hw
            baseline_uv = F.interpolate(
                baseline_uv_lr.reshape(B * K, Cuv, H, W),
                size=(oh, ow), mode="bilinear", align_corners=False,
            ).reshape(B, K, Cuv, oh, ow)
        else:
            baseline_uv = baseline_uv_lr
        output["uv"] = output["uv"] + baseline_uv

        return output
