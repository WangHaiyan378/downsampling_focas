import math
import torch
import torch.nn as nn


class LeadTimeEmbedding(nn.Module):
    """Sinusoidal + learned lead time embedding."""

    def __init__(self, d_model, max_lead=60):
        super().__init__()
        self.d_model = d_model
        self.max_lead = max_lead
        self.learned = nn.Embedding(max_lead + 1, d_model)
        pe = torch.zeros(max_lead + 1, d_model)
        position = torch.arange(0, max_lead + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, lead_ids):
        """
        lead_ids: (B, K) integer lead time indices
        returns: (B, K, d_model)
        """
        lead_ids = lead_ids.clamp(0, self.max_lead).long()
        sinusoidal = self.pe[lead_ids]
        learned = self.learned(lead_ids)
        return sinusoidal + learned


class SpatialLeadModulation(nn.Module):
    """Expand lead embedding to spatial dimensions for feature modulation."""

    def __init__(self, d_model):
        super().__init__()
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)

    def forward(self, features, lead_emb):
        """
        features: (B, C, H, W)
        lead_emb: (B, C) or (B, K, C) for single lead
        returns: modulated features
        """
        if lead_emb.dim() == 3:
            lead_emb = lead_emb.mean(dim=1)
        gamma = self.gamma_proj(lead_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(lead_emb).unsqueeze(-1).unsqueeze(-1)
        return features * (1 + gamma) + beta
