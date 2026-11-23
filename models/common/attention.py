import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, N, D)"""
        residual = x
        x = self.norm(x)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return residual + self.dropout(out)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        query: (B, Nq, D)
        context: (B, Nc, D)
        """
        residual = query
        query = self.norm_q(query)
        context = self.norm_kv(context)
        B, Nq, D = query.shape
        Nc = context.size(1)

        q = self.q_proj(query).reshape(B, Nq, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, Nc, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, Nc, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, D)
        out = self.out_proj(out)
        return residual + self.dropout(out)


class SpatialAttention(nn.Module):
    """Downsampled spatial self-attention for H×W feature maps."""

    def __init__(self, d_model, n_heads, spatial_downsample=4, dropout=0.1):
        super().__init__()
        self.ds = spatial_downsample
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        if self.ds > 1 and (H > self.ds * 2 and W > self.ds * 2):
            x_ds = F.adaptive_avg_pool2d(x, (H // self.ds, W // self.ds))
            Hd, Wd = x_ds.shape[2], x_ds.shape[3]
        else:
            x_ds = x
            Hd, Wd = H, W

        tokens = x_ds.flatten(2).permute(0, 2, 1)
        tokens = self.attn(tokens)
        tokens = tokens + self.ffn(tokens)
        out = tokens.permute(0, 2, 1).reshape(B, C, Hd, Wd)

        if Hd != H or Wd != W:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            out = out + x
        return out


class TemporalAttention(nn.Module):
    """Attention over the time dimension."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, T, D)"""
        x = self.attn(x)
        x = x + self.ffn(x)
        return x
