"""
Wavelet + LCZ multi-scale per-model forecast encoder for Forecast2ERA5FitModel.

主要組件
-----------------------------------------------------------------------------
WaveletTemporalDecomposition          沿 lead/time 維 K 做 Haar 多尺度分解
ScaleForecastEncoder                  每個小波尺度的 per-lead CNN 編碼 + lead 聚合
LCZTerrainEncoder                     靜態場 (LCZ / 地形 / lat-lon) 編碼，會自動 resize
ScaleLCZWindowCrossAttention2DRoPE    每個尺度上以 LCZ 作 K/V 的 window cross-attn，
                                      Q/K 上注入 2D RoPE
ScaleLCZGate                          以 raw forecast / lead emb / LCZ 統計生成 alpha
WaveletLCZForecastEncoder             把上述 5 個模組串成單一 per-model encoder

工具函數
-----------------------------------------------------------------------------
rotate_half / apply_1d_rope / build_2d_rope_cache / apply_2d_rope
pad_to_window_size / unpad_from_window_size
window_partition / window_reverse
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.conv_blocks import DoubleConv, ResBlock


# =============================================================================
# 工具：window partition / pad
# =============================================================================

def pad_to_window_size(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    x: [B, D, H, W]
    回傳 (x_pad, H_pad, W_pad)，pad 到能被 window_size 整除。
    """
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, H, W
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, H + pad_h, W + pad_w


def unpad_from_window_size(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """x: [B, D, H_pad, W_pad] → [B, D, H, W]"""
    return x[..., :H, :W].contiguous()


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: [B, D, H, W]  (H, W 必須能被 window_size 整除)
    return: [B*num_windows, ws*ws, D]
    """
    B, D, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "需要先 pad_to_window_size"
    nh = H // window_size
    nw = W // window_size
    x = x.view(B, D, nh, window_size, nw, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nh, nw, ws, ws, D]
    x = x.view(B * nh * nw, window_size * window_size, D)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """
    windows: [B*num_windows, ws*ws, D]
    return:  [B, D, H, W]
    """
    nh = H // window_size
    nw = W // window_size
    D = windows.shape[-1]
    x = windows.view(B, nh, nw, window_size, window_size, D)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, D, nh, ws, nw, ws]
    x = x.view(B, D, H, W)
    return x


# =============================================================================
# 工具：2D RoPE
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    x[..., 2i], x[..., 2i+1] → (-x[..., 2i+1], x[..., 2i])
    保持最後一維長度不變。要求最後一維為偶數。
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.stack((-x_odd, x_even), dim=-1)
    return out.flatten(-2)


def apply_1d_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [B, H, N, d]
    cos: [1, 1, N, d]
    sin: [1, 1, N, d]
    """
    return x * cos + rotate_half(x) * sin


def build_2d_rope_cache(
    window_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    rope_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    回傳 (cos_y, sin_y, cos_x, sin_x)，shape 都是 [1, 1, N, head_dim_half]，
    其中 N = window_size * window_size，head_dim_half = head_dim // 2。
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D RoPE 需要 head_dim % 4 == 0（目前 head_dim={head_dim}）。"
            "請調整 d_model 或 lcz_attn_heads，使 per-head dim 能被 4 整除，"
            "或在 cfg 中設 use_lcz_2d_rope: false。"
        )
    half = head_dim // 2  # y 與 x 各佔一半

    # 針對「每個 head_dim_half 內」的 2 維 pair 建頻率（同 1D RoPE）
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half))
    # inv_freq: [half // 2]

    pos = torch.arange(window_size, device=device, dtype=torch.float32)  # [ws]
    pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
    pos_y = pos_y.reshape(-1)  # [N]
    pos_x = pos_x.reshape(-1)  # [N]

    def _build(p):
        freqs = torch.einsum("n,d->nd", p, inv_freq)        # [N, half//2]
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # [N, half]
        cos = emb.cos().to(dtype)[None, None, :, :]         # [1,1,N,half]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin

    cos_y, sin_y = _build(pos_y)
    cos_x, sin_x = _build(pos_x)
    return cos_y, sin_y, cos_x, sin_x


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    window_size: int,
    rope_base: float = 10000.0,
    cache=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B_window, num_heads, N, head_dim]，N = ws*ws
    回傳 (q_rope, k_rope)，shape 不變。
    """
    head_dim = q.shape[-1]
    if cache is None:
        cos_y, sin_y, cos_x, sin_x = build_2d_rope_cache(
            window_size, head_dim, q.device, q.dtype, rope_base,
        )
    else:
        cos_y, sin_y, cos_x, sin_x = cache

    half = head_dim // 2
    q_y, q_x = q[..., :half], q[..., half:]
    k_y, k_x = k[..., :half], k[..., half:]

    q_y = apply_1d_rope(q_y, cos_y, sin_y)
    k_y = apply_1d_rope(k_y, cos_y, sin_y)
    q_x = apply_1d_rope(q_x, cos_x, sin_x)
    k_x = apply_1d_rope(k_x, cos_x, sin_x)

    q_rope = torch.cat([q_y, q_x], dim=-1)
    k_rope = torch.cat([k_y, k_x], dim=-1)
    return q_rope, k_rope


# =============================================================================
# 1) 小波多尺度時間分解 (K 維)
# =============================================================================

class WaveletTemporalDecomposition(nn.Module):
    """
    沿 K（lead/time）維做多層 Haar 小波分解，每個尺度都重構回長度 K 後輸出。

    輸入: [B, K, C, H, W]
    輸出: [B, S, K, C, H, W]
    """

    def __init__(
        self,
        levels: int = 2,
        wavelet_type: str = "haar",
        include_raw: bool = True,
    ):
        super().__init__()
        if wavelet_type.lower() != "haar":
            # 為避免 pywt 依賴，僅實作 Haar；其他類型 fallback 到 Haar 並 print warning
            print(f"[WaveletTemporalDecomposition] '{wavelet_type}' not supported in torch fallback; using 'haar'.")
        self.levels = max(int(levels), 1)
        self.include_raw = bool(include_raw)

    @staticmethod
    def _haar_decompose_one_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, K, C, H, W]，沿 K 做一層 Haar：
            low  = (x_even + x_odd)/sqrt(2)
            high = (x_even - x_odd)/sqrt(2)
        若 K 為奇數，會在尾端複製一次再分解，最後會在 _upsample 時 trim 回原長度。
        回傳: (low, high), shape 各為 [B, K//2 (或 (K+1)//2), C, H, W]
        """
        B, K, C, H, W = x.shape
        if K % 2 == 1:
            x = torch.cat([x, x[:, -1:].contiguous()], dim=1)
            K = K + 1
        x = x.view(B, K // 2, 2, C, H, W)
        even = x[:, :, 0]
        odd = x[:, :, 1]
        s = math.sqrt(2.0)
        low = (even + odd) / s
        high = (even - odd) / s
        return low, high

    @staticmethod
    def _upsample_to_K(x: torch.Tensor, K: int) -> torch.Tensor:
        """
        x: [B, K_short, C, H, W] → [B, K, C, H, W]
        用 repeat_interleave + 截斷對齊 Haar 的「兩個原點對應一個低頻點」尺度。
        """
        rep = x.repeat_interleave(2, dim=1)
        rep = rep[:, :K].contiguous()
        if rep.shape[1] < K:  # 理論上不會發生，但保險
            pad = K - rep.shape[1]
            rep = torch.cat([rep, rep[:, -1:].expand(-1, pad, *rep.shape[2:])], dim=1)
        return rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, C, H, W]
        return: [B, S, K, C, H, W]
        """
        B, K, C, H, W = x.shape
        scales = []

        if self.include_raw:
            scales.append(x)

        # 逐層分解：每層生 high (D_l)，最後一層保留 low (A_L)
        cur = x
        highs = []
        for _ in range(self.levels):
            low, high = self._haar_decompose_one_level(cur)
            highs.append(high)
            cur = low
        approx = cur  # 最深層的低頻

        scales.append(self._upsample_to_K(approx, K))  # 低頻 / 趨勢
        for h in highs:
            scales.append(self._upsample_to_K(h, K))   # 各層高頻 D1, D2, ...

        out = torch.stack(scales, dim=1)
        return out  # [B, S, K, C, H, W]


# =============================================================================
# 2) 單一尺度 forecast encoder（沿用原 ForecastEncoder 思路）
# =============================================================================

class ScaleForecastEncoder(nn.Module):
    """
    輸入: [B, K, C_in, H, W]   單一小波尺度
    輸出: [B, D, H, W]
    """

    def __init__(self, c_in: int, d_model: int, num_leads: int, chunk_k: int = 8):
        super().__init__()
        self.num_leads = num_leads
        self.chunk_k = int(chunk_k)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = x.shape
        assert K == self.num_leads, f"ScaleForecastEncoder expects K={self.num_leads}, got {K}"
        parts = []
        for k0 in range(0, K, self.chunk_k):
            k1 = min(k0 + self.chunk_k, K)
            xf = x[:, k0:k1].contiguous().reshape(-1, C, H, W)
            f = self.per_lead_conv(xf)
            parts.append(f.reshape(B, k1 - k0, f.shape[1], H, W))
        feats = torch.cat(parts, dim=1)
        D = feats.shape[2]
        feats = feats.reshape(B, K * D, H, W)
        return self.lead_aggregator(feats)  # [B, D, H, W]


# =============================================================================
# 3) LCZ / 靜態場 encoder（自動 resize 到 forecast 空間大小）
# =============================================================================

class LCZTerrainEncoder(nn.Module):
    """
    輸入: x_static [B, C_static, H_s, W_s], target_hw=(H, W)
    輸出: [B, D, H, W]

    預設只用連續特徵；若想 embedding LCZ id，呼叫端可在 cfg 中自行擴充包裝。
    """

    def __init__(self, c_static: int, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_static, d_model // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, d_model // 2), d_model // 2),
            nn.GELU(),
            ResBlock(d_model // 2),
            nn.Conv2d(d_model // 2, d_model, 1),
        )

    def forward(self, x_static: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = target_hw
        if x_static.shape[-2:] != (H, W):
            x_static = F.interpolate(x_static, size=(H, W), mode="bilinear", align_corners=False)
        return self.encoder(x_static)


# =============================================================================
# 4) Scale × LCZ window cross-attention（Q=尺度特徵, K/V=LCZ）+ 2D RoPE
# =============================================================================

class ScaleLCZWindowCrossAttention2DRoPE(nn.Module):
    """
    輸入:
        scale_feats: [B, S, D, H, W]
        lcz_feat:    [B, D, H, W]
    輸出:
        scale_feats_lcz: [B, S, D, H, W]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.1,
        use_2d_rope: bool = True,
        rope_base: float = 10000.0,
        ffn_mult: int = 2,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必須能被 num_heads={num_heads} 整除")
        head_dim = d_model // num_heads
        if use_2d_rope and head_dim % 4 != 0:
            raise ValueError(
                f"use_lcz_2d_rope=True 需要 per-head dim 能被 4 整除，"
                f"目前 d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}。"
                f"請調整 lcz_attn_heads（例如改為 {num_heads * 2}）或 d_model。"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = int(window_size)
        self.use_2d_rope = bool(use_2d_rope)
        self.rope_base = float(rope_base)
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

        # RoPE cache（依 device/dtype lazy 構建）
        self._rope_cache_key: Optional[Tuple] = None
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def _get_rope_cache(self, device, dtype):
        if not self.use_2d_rope:
            return None
        key = (self.window_size, self.head_dim, device, dtype)
        if self._rope_cache_key != key:
            self._rope_cache = build_2d_rope_cache(
                self.window_size, self.head_dim, device, dtype, self.rope_base,
            )
            self._rope_cache_key = key
        return self._rope_cache

    def _attn_one_scale(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        """
        q_feat:  [B, D, H, W]   (scale wind feature)
        kv_feat: [B, D, H, W]   (LCZ / static feature)
        return:  [B, D, H, W]
        """
        B, D, H, W = q_feat.shape
        ws = self.window_size

        q_pad, Hp, Wp = pad_to_window_size(q_feat, ws)
        kv_pad, _, _ = pad_to_window_size(kv_feat, ws)

        q_win = window_partition(q_pad, ws)        # [B*nW, ws*ws, D]
        kv_win = window_partition(kv_pad, ws)
        Bw, N, _ = q_win.shape

        q_in = self.norm_q(q_win)
        kv_in = self.norm_kv(kv_win)

        q = self.q_proj(q_in).view(Bw, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).view(Bw, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).view(Bw, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q,k,v: [Bw, heads, N, head_dim]

        if self.use_2d_rope:
            cache = self._get_rope_cache(q.device, q.dtype)
            q, k = apply_2d_rope(q, k, ws, self.rope_base, cache=cache)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)               # [Bw, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(Bw, N, D)

        out = self.out_proj(out)
        out = self.proj_drop(out)
        out = q_win + out                          # residual at window level

        out = out + self.ffn(self.norm_ffn(out))

        x = window_reverse(out, ws, Hp, Wp, B)     # [B, D, Hp, Wp]
        x = unpad_from_window_size(x, H, W)        # [B, D, H, W]
        return x

    def forward(self, scale_feats: torch.Tensor, lcz_feat: torch.Tensor) -> torch.Tensor:
        B, S, D, H, W = scale_feats.shape
        outs = []
        for s in range(S):
            outs.append(self._attn_one_scale(scale_feats[:, s], lcz_feat))
        return torch.stack(outs, dim=1)  # [B, S, D, H, W]


# =============================================================================
# 5) Scale-LCZ gate：根據 raw forecast/lead/LCZ 統計輸出 alpha
# =============================================================================

class ScaleLCZGate(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_scales: int,
        c_in: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or d_model)
        self.raw_proj = nn.Linear(2 * c_in, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scales),
        )
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        raw_forecast: torch.Tensor,         # [B, K, C_in, H, W]
        scale_feats_lcz: torch.Tensor,      # [B, S, D, H, W]
        lead_emb: torch.Tensor,             # [B, K, D]
        lcz_feat: torch.Tensor,             # [B, D, H, W]
    ) -> torch.Tensor:
        # raw 統計：對 K, H, W 取 mean / std → [B, C_in]
        raw_mean = raw_forecast.mean(dim=(1, 3, 4))
        raw_std = raw_forecast.std(dim=(1, 3, 4))
        raw_pool = torch.cat([raw_mean, raw_std], dim=-1)        # [B, 2*C_in]
        raw_pool = self.raw_proj(raw_pool)                        # [B, D]

        lead_pool = lead_emb.mean(dim=1)                          # [B, D]
        lcz_pool = lcz_feat.mean(dim=(-2, -1))                    # [B, D]
        scale_pool = scale_feats_lcz.mean(dim=(-2, -1)).mean(dim=1)  # [B, D]

        gate_in = torch.cat([raw_pool, lead_pool, lcz_pool, scale_pool], dim=-1)
        logits = self.mlp(gate_in)
        alpha = logits.softmax(dim=-1)                            # [B, S]
        self.last_alpha = alpha.detach()
        return alpha


# =============================================================================
# 6) WaveletLCZForecastEncoder：頂層 per-model encoder
# =============================================================================

class WaveletLCZForecastEncoder(nn.Module):
    """
    取代原本的 ForecastEncoder。

    輸入:
        x_m:      [B, K, C_in, H, W]
        x_static: [B, C_static, H_s, W_s]
        lead_emb: [B, K, D]
    輸出:
        model_feat: [B, D, H, W]
    """

    def __init__(
        self,
        c_in: int,
        c_static: int,
        d_model: int,
        num_leads: int,
        wavelet_levels: int = 2,
        wavelet_include_raw: bool = True,
        wavelet_type: str = "haar",
        use_lcz_window_cross_attention: bool = True,
        use_lcz_2d_rope: bool = True,
        lcz_attn_window_size: int = 8,
        lcz_attn_heads: int = 4,
        lcz_attn_dropout: Optional[float] = None,
        lcz_rope_base: float = 10000.0,
        use_lcz_scale_gate: bool = True,
        scale_gate_hidden_dim: Optional[int] = None,
        scale_gate_dropout: Optional[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_leads = num_leads
        self.use_lcz_attn = bool(use_lcz_window_cross_attention)
        self.use_lcz_gate = bool(use_lcz_scale_gate)

        self.wavelet_decomp = WaveletTemporalDecomposition(
            levels=wavelet_levels,
            wavelet_type=wavelet_type,
            include_raw=wavelet_include_raw,
        )
        # 計算實際小波尺度數 S
        self.num_scales = wavelet_levels + (1 if wavelet_include_raw else 0) + 1  # +1 for approx (low)

        self.scale_encoders = nn.ModuleList([
            ScaleForecastEncoder(c_in, d_model, num_leads)
            for _ in range(self.num_scales)
        ])

        self.lcz_encoder = LCZTerrainEncoder(c_static, d_model)

        if self.use_lcz_attn:
            self.scale_lcz_attn = ScaleLCZWindowCrossAttention2DRoPE(
                d_model=d_model,
                num_heads=lcz_attn_heads,
                window_size=lcz_attn_window_size,
                dropout=(dropout if lcz_attn_dropout is None else lcz_attn_dropout),
                use_2d_rope=use_lcz_2d_rope,
                rope_base=lcz_rope_base,
            )
        else:
            self.scale_lcz_attn = None

        if self.use_lcz_gate:
            self.scale_gate = ScaleLCZGate(
                d_model=d_model,
                num_scales=self.num_scales,
                c_in=c_in,
                hidden_dim=scale_gate_hidden_dim,
                dropout=(dropout if scale_gate_dropout is None else scale_gate_dropout),
            )
        else:
            # 無 gate 時退化為等權平均
            self.scale_gate = None

    def forward(
        self,
        x_m: torch.Tensor,                  # [B, K, C_in, H, W]
        x_static: torch.Tensor,             # [B, C_static, H_s, W_s]
        lead_emb: torch.Tensor,             # [B, K, D]
    ) -> torch.Tensor:
        B, K, C, H, W = x_m.shape

        scale_components = self.wavelet_decomp(x_m)  # [B, S, K, C, H, W]
        S = scale_components.shape[1]
        assert S == self.num_scales, (
            f"WaveletTemporalDecomposition 出 S={S}，但 encoder 預期 S={self.num_scales}。"
            "請檢查 wavelet_levels / wavelet_include_raw 設定。"
        )

        scale_feats = []
        for s in range(S):
            scale_feats.append(self.scale_encoders[s](scale_components[:, s]))
        scale_feats = torch.stack(scale_feats, dim=1)  # [B, S, D, H, W]

        lcz_feat = self.lcz_encoder(x_static, target_hw=(H, W))  # [B, D, H, W]

        if self.scale_lcz_attn is not None:
            scale_feats_lcz = self.scale_lcz_attn(scale_feats, lcz_feat)
        else:
            scale_feats_lcz = scale_feats

        if self.scale_gate is not None:
            alpha = self.scale_gate(x_m, scale_feats_lcz, lead_emb, lcz_feat)  # [B, S]
        else:
            alpha = scale_feats_lcz.new_full((B, S), 1.0 / S)

        # alpha: [B, S] → [B, S, 1, 1, 1]，與 scale_feats_lcz 相乘後在 S 上 sum
        model_feat = (scale_feats_lcz * alpha[:, :, None, None, None]).sum(dim=1)
        return model_feat  # [B, D, H, W]
