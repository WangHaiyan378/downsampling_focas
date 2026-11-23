import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """2D Haar wavelet decomposition and reconstruction for spatial multi-scale analysis."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("dec_filters", filters)

        inv_ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        inv_lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        inv_hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        inv_hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        rec_filters = torch.stack([inv_ll, inv_lh, inv_hl, inv_hh], dim=0).unsqueeze(1)
        self.register_buffer("rec_filters", rec_filters)

    def decompose(self, x):
        """
        Decompose input into 4 subbands: LL, LH, HL, HH.
        x: (B, C, H, W) -> returns (B, 4*C, H//2, W//2)
        """
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        coeffs = F.conv2d(x, self.dec_filters, stride=2)
        _, _, Hc, Wc = coeffs.shape
        coeffs = coeffs.reshape(B, C, 4, Hc, Wc)
        coeffs = coeffs.reshape(B, C * 4, Hc, Wc)
        return coeffs

    def reconstruct(self, coeffs, target_h, target_w):
        """
        Reconstruct from subbands.
        coeffs: (B, 4*C, H//2, W//2) -> (B, C, H, W)
        """
        B, C4, Hc, Wc = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.reshape(B * C, 4, Hc, Wc)
        rec = torch.zeros(B * C, 1, Hc * 2, Wc * 2, device=coeffs.device)
        for i in range(4):
            rec += F.conv_transpose2d(
                coeffs[:, i : i + 1], self.rec_filters[i : i + 1], stride=2
            )
        rec = rec.reshape(B, C, Hc * 2, Wc * 2)
        if rec.shape[2] != target_h or rec.shape[3] != target_w:
            rec = rec[:, :, :target_h, :target_w]
        return rec
