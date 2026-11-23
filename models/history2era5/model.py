"""
Task A: History2ERA5 Model
Input: Multi-model historical sequences [B, M, T, C_in, H, W]
       Static features [B, C_static, H, W]
       Lead IDs [B, K]
Output: Future ERA5 uv + convergence at K lead times

Architecture:
1. Per-model dual-branch encoder (raw + wavelet)
2. Temporal history encoding via attention
3. LCZ / terrain FiLM conditioning
4. Multi-model dynamic fusion via cross-attention
5. Temporal-Scale Dynamics Unit (TSDU)
6. Lead-aware output head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.wavelet import HaarWavelet2D
from models.common.conv_blocks import ResBlock, DoubleConv, MultiScaleConv
from models.common.attention import (
    SpatialAttention,
    TemporalAttention,
    CrossAttention,
    MultiHeadSelfAttention,
)
from models.common.conditioning import TerrainLCZConditioner
from models.common.lead_embed import LeadTimeEmbedding, SpatialLeadModulation
from models.common.output_head import UVConvOutputHead


class DualBranchEncoder(nn.Module):
    """Encode one model's single timestep with raw + wavelet branches."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.raw_branch = nn.Sequential(
            DoubleConv(c_in, d_model // 2),
            ResBlock(d_model // 2),
        )
        self.wavelet = HaarWavelet2D()
        self.wavelet_branch = nn.Sequential(
            DoubleConv(c_in * 4, d_model // 2),
            ResBlock(d_model // 2),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.GELU(),
        )

    def forward(self, x):
        """x: (B, C_in, H, W) -> (B, D, H, W)"""
        raw_feat = self.raw_branch(x)

        w_coeffs = self.wavelet.decompose(x)
        Hc, Wc = w_coeffs.shape[2], w_coeffs.shape[3]
        wav_feat = self.wavelet_branch(w_coeffs)
        wav_feat = F.interpolate(wav_feat, size=raw_feat.shape[2:], mode="bilinear", align_corners=False)

        merged = torch.cat([raw_feat, wav_feat], dim=1)
        return self.merge(merged)


class PerModelHistoryEncoder(nn.Module):
    """Encode history for a single model: [B, T, C_in, H, W] -> [B, D, H, W]"""

    def __init__(self, c_in, d_model, n_heads, history_len, n_layers=2, dropout=0.1):
        super().__init__()
        self.dual_encoder = DualBranchEncoder(c_in, d_model)
        self.temporal_attn = nn.ModuleList([
            TemporalAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, T, C_in, H, W)
        returns: (B, D, H, W) aggregated history features
        """
        B, T, C, H, W = x.shape
        # 分塊通過 dual_encoder，避免 T=45 時 B*T 一次性卷積佔滿顯存
        chunk_t = 8
        parts = []
        for t0 in range(0, T, chunk_t):
            t1 = min(t0 + chunk_t, T)
            xf = x[:, t0:t1].contiguous().reshape(-1, C, H, W)
            f = self.dual_encoder(xf)
            parts.append(f.reshape(B, t1 - t0, f.shape[1], H, W))
        feats_spatial = torch.cat(parts, dim=1)
        D = feats_spatial.shape[2]

        pooled = self.spatial_pool(feats_spatial.reshape(B * T, D, H, W))
        pooled = pooled.reshape(B, T, D)

        for attn in self.temporal_attn:
            pooled = attn(pooled)

        temporal_weights = F.softmax(self.temporal_proj(pooled), dim=1)
        temporal_weights = temporal_weights.unsqueeze(-1).unsqueeze(-1)

        feats_spatial = feats_spatial * temporal_weights
        aggregated = feats_spatial.sum(dim=1)

        return aggregated


class MultiModelFusion(nn.Module):
    """Dynamically fuse features from M models using attention."""

    def __init__(self, d_model, n_heads, n_layers=2, dropout=0.1):
        super().__init__()
        self.model_attn = nn.ModuleList([
            MultiHeadSelfAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, model_features):
        """
        model_features: (B, M, D, H, W)
        returns: (B, D, H, W) fused features
        """
        B, M, D, H, W = model_features.shape
        spatial_pool = model_features.mean(dim=(-2, -1))

        for attn in self.model_attn:
            spatial_pool = attn(spatial_pool)

        weights = self.gate(spatial_pool)
        weights = F.softmax(weights, dim=1)

        fused_pool = (spatial_pool * weights).sum(dim=1)
        fused_pool = fused_pool + self.ffn(fused_pool)

        weights_expand = weights.unsqueeze(-1).unsqueeze(-1)
        fused_spatial = (model_features * weights_expand).sum(dim=1)

        return fused_spatial


class TSDU(nn.Module):
    """
    Temporal-Scale Dynamics Unit:
    Multi-scale temporal dynamics with dilated convolutions and attention.
    Operates on spatial features conditioned by lead embeddings.
    """

    def __init__(self, d_model, n_heads, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ms_conv": MultiScaleConv(d_model, dilations=(1, 2, 4)),
                "spatial_attn": SpatialAttention(d_model, n_heads, spatial_downsample=4, dropout=dropout),
                "res": ResBlock(d_model, dropout=dropout),
            }))
        self.norm = nn.GroupNorm(min(32, d_model), d_model)

    def forward(self, x, lead_modulation=None):
        """
        x: (B, D, H, W)
        lead_modulation: optional (B, D, H, W) additive modulation
        returns: (B, D, H, W)
        """
        if lead_modulation is not None:
            x = x + lead_modulation
        for layer in self.layers:
            x = layer["ms_conv"](x)
            x = layer["spatial_attn"](x)
            x = layer["res"](x)
        return self.norm(x)


class History2ERA5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_models = cfg["num_models"]
        self.num_leads = cfg["num_leads"]
        d = cfg["d_model"]
        c_in = cfg["c_in"]
        c_static = cfg["c_static"]
        n_heads = cfg["n_heads"]
        history_len = cfg["history_len"]
        dropout = cfg.get("dropout", 0.1)

        enc_h = int(cfg.get("encoder_grid_h", cfg.get("grid_h", 121)))
        enc_w = int(cfg.get("encoder_grid_w", cfg.get("grid_w", 240)))
        out_h = int(cfg.get("grid_h", enc_h))
        out_w = int(cfg.get("grid_w", enc_w))
        self._enc_hw = (enc_h, enc_w)
        self._out_hw = (out_h, out_w)
        self._needs_spatial_up = (enc_h, enc_w) != (out_h, out_w)

        self.per_model_encoders = nn.ModuleList([
            PerModelHistoryEncoder(
                c_in, d, n_heads, history_len,
                n_layers=cfg.get("n_encoder_layers", 3),
                dropout=dropout,
            )
            for _ in range(self.num_models)
        ])

        self.conditioner = TerrainLCZConditioner(c_static, d)

        self.lead_embed = LeadTimeEmbedding(d, max_lead=120)
        self.lead_spatial = SpatialLeadModulation(d)

        self.fusion = MultiModelFusion(
            d, n_heads,
            n_layers=cfg.get("n_fusion_layers", 2),
            dropout=dropout,
        )

        self.tsdu = TSDU(
            d, n_heads,
            n_layers=cfg.get("n_tsdu_layers", 3),
            dropout=dropout,
        )

        self.lead_to_spatial = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
        )

        self.spatial_up = None
        if self._needs_spatial_up:
            # 1x1 only: 全 ERA5 格點上 3x3+GN 峰值記憶體過大，易觸發 OOM
            self.spatial_up = nn.Conv2d(d, d, kernel_size=1)

        self.output_head = UVConvOutputHead(
            d, self.num_leads,
            c_out_uv=cfg.get("c_out_uv", 2),
            c_out_conv=cfg.get("c_out_conv", 1),
            grid_h=out_h,
            grid_w=out_w,
        )

    def forward(self, x_history, x_static, lead_ids):
        """
        x_history: (B, M, T, C_in, H, W)
        x_static: (B, C_static, H, W)
        lead_ids: (B, K)
        """
        B, M, T, C, H, W = x_history.shape

        model_feats = []
        for m in range(M):
            feat = self.per_model_encoders[m](x_history[:, m])
            model_feats.append(feat)
        model_feats = torch.stack(model_feats, dim=1)

        model_feats_cond = []
        for m in range(M):
            cond_feat = self.conditioner(model_feats[:, m], x_static)
            model_feats_cond.append(cond_feat)
        model_feats = torch.stack(model_feats_cond, dim=1)

        fused = self.fusion(model_feats)

        lead_emb = self.lead_embed(lead_ids)
        lead_spatial = self.lead_to_spatial(lead_emb.mean(dim=1))
        lead_mod = lead_spatial.unsqueeze(-1).unsqueeze(-1).expand_as(fused)

        dynamics = self.tsdu(fused, lead_mod)

        lead_for_mod = self.lead_embed(lead_ids)
        dynamics = self.lead_spatial(dynamics, lead_for_mod)

        if self._needs_spatial_up:
            oh, ow = self._out_hw
            dynamics = F.interpolate(
                dynamics, size=(oh, ow), mode="bilinear", align_corners=False,
            )
            dynamics = self.spatial_up(dynamics)

        output = self.output_head(dynamics)
        return output
