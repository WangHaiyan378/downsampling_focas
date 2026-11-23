"""
Unified loss framework for both tasks:
- UV loss (MSE + MAE)
- Conv loss (convergence prediction)
- Conv consistency loss (direct vs diagnostic)
- Direction loss (wind direction)
- Gradient loss (spatial smoothness)
- Lead-dependent weighting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_uv, target_uv):
        """
        pred_uv: (B, K, 2, H, W)
        target_uv: (B, K, 2, H, W)
        returns: per-lead loss (B, K)
        """
        mse = ((pred_uv - target_uv) ** 2).mean(dim=(-3, -2, -1))
        mae = (pred_uv - target_uv).abs().mean(dim=(-3, -2, -1))
        return 0.7 * mse + 0.3 * mae


class ConvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_conv, target_conv):
        """
        pred_conv: (B, K, 1, H, W)
        target_conv: (B, K, 1, H, W)
        """
        return ((pred_conv - target_conv) ** 2).mean(dim=(-3, -2, -1))


class ConvConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, direct_conv, diag_conv):
        """Penalize difference between directly predicted and diagnostic convergence."""
        return ((direct_conv - diag_conv) ** 2).mean(dim=(-3, -2, -1))


class DirectionLoss(nn.Module):
    def __init__(self, eps: float = 1.0e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred_uv, target_uv):
        """
        Cosine-based direction loss on wind vectors, numerically safe under AMP.
        pred_uv, target_uv: (B, K, 2, H, W). Inputs expected to be float32.
        """
        pred_u = pred_uv[:, :, 0]
        pred_v = pred_uv[:, :, 1]
        target_u = target_uv[:, :, 0]
        target_v = target_uv[:, :, 1]

        eps = self.eps
        pred_speed = torch.sqrt(pred_u * pred_u + pred_v * pred_v + eps).clamp_min(eps)
        target_speed = torch.sqrt(target_u * target_u + target_v * target_v + eps).clamp_min(eps)

        dot = pred_u * target_u + pred_v * target_v
        cos_sim = dot / (pred_speed * target_speed)
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        direction_loss = 1.0 - cos_sim
        return direction_loss.mean(dim=(-1, -2))


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_uv, target_uv):
        """
        Spatial gradient loss: penalize differences in spatial derivatives.
        """
        pred_dx = pred_uv[:, :, :, :, 1:] - pred_uv[:, :, :, :, :-1]
        target_dx = target_uv[:, :, :, :, 1:] - target_uv[:, :, :, :, :-1]
        pred_dy = pred_uv[:, :, :, 1:, :] - pred_uv[:, :, :, :-1, :]
        target_dy = target_uv[:, :, :, 1:, :] - target_uv[:, :, :, :-1, :]

        loss_dx = ((pred_dx - target_dx) ** 2).mean(dim=(-3, -2, -1))
        loss_dy = ((pred_dy - target_dy) ** 2).mean(dim=(-3, -2, -1))
        return loss_dx + loss_dy


class UnifiedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg["loss"]
        self.uv_weight = loss_cfg.get("uv_weight", 1.0)
        self.conv_weight = loss_cfg.get("conv_weight", 0.5)
        self.conv_consistency_weight = loss_cfg.get("conv_consistency_weight", 0.3)
        self.direction_weight = loss_cfg.get("direction_weight", 0.2)
        self.gradient_weight = loss_cfg.get("gradient_weight", 0.1)
        self.lead_decay = loss_cfg.get("lead_decay", 0.95)

        mcfg = cfg["model"]
        gh = int(mcfg.get("grid_h", 121))
        gw = int(mcfg.get("grid_w", 240))
        dlat = 180.0 / max(gh - 1, 1)
        dlon = 360.0 / max(gw, 1)
        self._conv_dx = float(dlon * 111000.0)
        self._conv_dy = float(dlat * 111000.0)

        self.uv_loss = UVLoss()
        self.conv_loss = ConvLoss()
        self.conv_consistency_loss = ConvConsistencyLoss()
        self.direction_loss = DirectionLoss()
        self.gradient_loss = GradientLoss()

    def compute_lead_weights(self, num_leads, device):
        """Exponentially decaying weights: shorter leads get higher weight."""
        indices = torch.arange(num_leads, dtype=torch.float32, device=device)
        weights = self.lead_decay ** indices
        weights = weights / weights.sum() * num_leads
        return weights

    def compute_target_conv(self, target_uv, dx=1.0, dy=1.0):
        """Compute convergence from target u, v."""
        u = target_uv[:, :, 0:1, :, :]
        v = target_uv[:, :, 1:2, :, :]

        dudx = torch.zeros_like(u)
        dudx[:, :, :, :, 1:-1] = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2.0 * dx)
        dudx[:, :, :, :, 0] = (u[:, :, :, :, 1] - u[:, :, :, :, 0]) / dx
        dudx[:, :, :, :, -1] = (u[:, :, :, :, -1] - u[:, :, :, :, -2]) / dx

        dvdy = torch.zeros_like(v)
        dvdy[:, :, :, 1:-1, :] = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2.0 * dy)
        dvdy[:, :, :, 0, :] = (v[:, :, :, 1, :] - v[:, :, :, 0, :]) / dy
        dvdy[:, :, :, -1, :] = (v[:, :, :, -1, :] - v[:, :, :, -2, :]) / dy

        return -(dudx + dvdy)

    def forward(self, pred, target_uv):
        """
        pred: dict with keys 'uv', 'direct_conv', 'diag_conv'
            uv: (B, K, 2, H, W)
            direct_conv: (B, K, 1, H, W)
            diag_conv: (B, K, 1, H, W)
        target_uv: (B, K, 2, H, W)

        All internal math runs in float32 to avoid FP16/BF16 underflow in
        sqrt / division, which otherwise causes NaN losses under AMP.
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_uv = pred["uv"].float()
            target_uv = target_uv.float()

            B, K = pred_uv.shape[:2]
            Cuv = int(pred_uv.shape[2])
            device = pred_uv.device

            has_conv_head = pred.get("direct_conv") is not None
            can_do_direction = Cuv >= 2
            can_do_conv_target = Cuv >= 2  # 需要 (u, v) 才能算 -(du/dx + dv/dy)

            lead_w = self.compute_lead_weights(K, device)

            l_uv = self.uv_loss(pred_uv, target_uv)
            l_grad = self.gradient_loss(pred_uv, target_uv)

            zero_scalar = torch.zeros((), device=device, dtype=torch.float32)
            l_conv = zero_scalar
            l_consist = zero_scalar
            l_dir = zero_scalar

            if has_conv_head and can_do_conv_target:
                direct_conv = pred["direct_conv"].float()
                diag_conv = pred["diag_conv"].float()
                target_conv = self.compute_target_conv(
                    target_uv, dx=self._conv_dx, dy=self._conv_dy,
                )
                l_conv = self.conv_loss(direct_conv, target_conv)
                l_consist = self.conv_consistency_loss(direct_conv, diag_conv)
                l_conv = (l_conv * lead_w.unsqueeze(0)).mean()
                l_consist = (l_consist * lead_w.unsqueeze(0)).mean()

            if can_do_direction:
                l_dir = self.direction_loss(pred_uv, target_uv)
                l_dir = (l_dir * lead_w.unsqueeze(0)).mean()

            l_uv = (l_uv * lead_w.unsqueeze(0)).mean()
            l_grad = (l_grad * lead_w.unsqueeze(0)).mean()

            # 對應權重若無此 component，乘以 0，保持 total 數值穩定且保留廣播語意
            w_conv = self.conv_weight if (has_conv_head and can_do_conv_target) else 0.0
            w_consist = self.conv_consistency_weight if (has_conv_head and can_do_conv_target) else 0.0
            w_dir = self.direction_weight if can_do_direction else 0.0

            total = (
                self.uv_weight * l_uv
                + w_conv * l_conv
                + w_consist * l_consist
                + w_dir * l_dir
                + self.gradient_weight * l_grad
            )

            components = {
                "total": total,
                "uv": l_uv,
                "conv": l_conv,
                "conv_consistency": l_consist,
                "direction": l_dir,
                "gradient": l_grad,
            }
            return total, components
