import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act=True, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            ConvBlock(in_ch, mid_ch),
            ConvBlock(mid_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels, hidden_ch=None, dropout=0.1):
        super().__init__()
        hidden_ch = hidden_ch or channels
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_ch, 3, 1, 1),
            nn.GroupNorm(min(32, hidden_ch), hidden_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_ch, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiScaleConv(nn.Module):
    """Multi-scale convolution with dilated branches."""

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        n_branches = len(dilations)
        branch_ch = channels // n_branches
        self.branches = nn.ModuleList()
        for d in dilations:
            ng = self._find_num_groups(branch_ch)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, branch_ch, 3, padding=d, dilation=d),
                    nn.GroupNorm(ng, branch_ch),
                    nn.GELU(),
                )
            )
        total_branch_ch = branch_ch * n_branches
        self.proj = nn.Conv2d(total_branch_ch, channels, 1)

    @staticmethod
    def _find_num_groups(ch):
        for g in [8, 4, 2, 1]:
            if ch % g == 0 and ch >= g:
                return g
        return 1

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        return x + self.proj(out)
