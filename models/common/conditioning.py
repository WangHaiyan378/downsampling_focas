import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation: condition features on static fields."""

    def __init__(self, feature_ch, cond_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(cond_ch, feature_ch, 3, padding=1),
            nn.GroupNorm(min(32, feature_ch), feature_ch),
            nn.GELU(),
            nn.Conv2d(feature_ch, feature_ch, 3, padding=1),
            nn.GroupNorm(min(32, feature_ch), feature_ch),
            nn.GELU(),
        )
        self.gamma_conv = nn.Conv2d(feature_ch, feature_ch, 1)
        self.beta_conv = nn.Conv2d(feature_ch, feature_ch, 1)

    def forward(self, features, condition):
        """
        features: (B, C, H, W)
        condition: (B, C_cond, H, W)
        """
        if condition.shape[2:] != features.shape[2:]:
            condition = F.interpolate(
                condition, size=features.shape[2:], mode="bilinear", align_corners=False
            )
        c = self.encoder(condition)
        gamma = self.gamma_conv(c)
        beta = self.beta_conv(c)
        return features * (1.0 + gamma) + beta


class TerrainLCZConditioner(nn.Module):
    """
    Multi-level terrain + LCZ conditioning using FiLM.
    Static features: terrain height, terrain slope, LCZ class, lat/lon encodings.
    """

    def __init__(self, c_static, d_model):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Conv2d(c_static, d_model // 2, 3, padding=1),
            nn.GroupNorm(min(32, d_model // 2), d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, 3, padding=1),
            nn.GroupNorm(min(32, d_model), d_model),
            nn.GELU(),
        )
        self.film = FiLMConditioning(d_model, d_model)

    def forward(self, features, static_features):
        """
        features: (B, D, H, W)
        static_features: (B, C_static, H, W)
        """
        cond = self.static_encoder(static_features)
        return self.film(features, cond)
