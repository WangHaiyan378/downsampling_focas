import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import ResBlock


class UVConvOutputHead(nn.Module):
    """
    Output head producing:
    - uv: (B, K, 2, H, W) - u10, v10 wind components
    - direct_conv: (B, K, 1, H, W) - directly predicted convergence
    - diag_conv: (B, K, 1, H, W) - diagnostic convergence from du/dx + dv/dy
    """

    def __init__(self, d_model, num_leads, c_out_uv=2, c_out_conv=1, grid_h=121, grid_w=240):
        super().__init__()
        self.num_leads = num_leads
        self.c_out_uv = int(c_out_uv)
        self.c_out_conv = int(c_out_conv)
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.enable_conv_head = self.c_out_conv > 0
        self.enable_diag_conv = self.c_out_uv >= 2

        npx = int(grid_h) * int(grid_w)
        # 全 ERA5 格點 + 多 lead 時，雙 ResBlock 與診斷輻合的峰值記憶體過大
        self._high_res = npx > 400_000
        if self._high_res:
            self.uv_head = nn.Sequential(
                nn.GroupNorm(min(32, d_model), d_model),
                nn.GELU(),
                nn.Conv2d(d_model, num_leads * self.c_out_uv, 1),
            )
            if self.enable_conv_head:
                self.conv_head = nn.Sequential(
                    nn.GroupNorm(min(32, d_model), d_model),
                    nn.GELU(),
                    nn.Conv2d(d_model, num_leads * self.c_out_conv, 1),
                )
            else:
                self.conv_head = None
        else:
            self.uv_head = nn.Sequential(
                ResBlock(d_model),
                ResBlock(d_model),
                nn.GroupNorm(min(32, d_model), d_model),
                nn.GELU(),
                nn.Conv2d(d_model, num_leads * self.c_out_uv, 1),
            )
            if self.enable_conv_head:
                self.conv_head = nn.Sequential(
                    ResBlock(d_model),
                    nn.GroupNorm(min(32, d_model), d_model),
                    nn.GELU(),
                    nn.Conv2d(d_model, num_leads * self.c_out_conv, 1),
                )
            else:
                self.conv_head = None

        dlat = 180.0 / (grid_h - 1) if grid_h > 1 else 1.0
        dlon = 360.0 / grid_w if grid_w > 0 else 1.0
        self.register_buffer("dx", torch.tensor(dlon * 111000.0))
        self.register_buffer("dy", torch.tensor(dlat * 111000.0))

    def compute_diagnostic_conv(self, uv):
        """
        Compute -(du/dx + dv/dy) from predicted wind components using finite differences.
        uv: (B, K, 2, H, W)
        """
        B, K, _, H, W = uv.shape
        if not self._high_res or K <= 8:
            return self._diagnostic_conv_chunk(uv)

        outs = []
        for k0 in range(0, K, 8):
            k1 = min(k0 + 8, K)
            outs.append(self._diagnostic_conv_chunk(uv[:, k0:k1]))
        return torch.cat(outs, dim=1)

    def _diagnostic_conv_chunk(self, uv):
        u = uv[:, :, 0:1, :, :]
        v = uv[:, :, 1:2, :, :]

        dudx = torch.zeros_like(u)
        dudx[:, :, :, :, 1:-1] = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2.0 * self.dx)
        dudx[:, :, :, :, 0] = (u[:, :, :, :, 1] - u[:, :, :, :, 0]) / self.dx
        dudx[:, :, :, :, -1] = (u[:, :, :, :, -1] - u[:, :, :, :, -2]) / self.dx

        dvdy = torch.zeros_like(v)
        dvdy[:, :, :, 1:-1, :] = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2.0 * self.dy)
        dvdy[:, :, :, 0, :] = (v[:, :, :, 1, :] - v[:, :, :, 0, :]) / self.dy
        dvdy[:, :, :, -1, :] = (v[:, :, :, -1, :] - v[:, :, :, -2, :]) / self.dy

        return -(dudx + dvdy)

    def forward(self, features):
        """
        features: (B, D, H, W) - fused feature map
        returns dict with:
            uv: (B, K, c_out_uv, H, W) — primary prediction (1 or 2+ channels)
            direct_conv / diag_conv: present only when conv head is enabled AND uv has ≥2 channels
        """
        B, D, H, W = features.shape
        K = self.num_leads

        uv_raw = self.uv_head(features)
        uv = uv_raw.reshape(B, K, self.c_out_uv, H, W)

        out = {"uv": uv}
        if self.enable_conv_head:
            conv_raw = self.conv_head(features)
            direct_conv = conv_raw.reshape(B, K, self.c_out_conv, H, W)
            out["direct_conv"] = direct_conv
            if self.enable_diag_conv:
                out["diag_conv"] = self.compute_diagnostic_conv(uv)
            else:
                out["diag_conv"] = torch.zeros_like(direct_conv)
        else:
            out["direct_conv"] = None
            out["diag_conv"] = None

        return out
