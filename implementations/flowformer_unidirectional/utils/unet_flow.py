import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.time_embedding import SinusoidalTimeEmbedding, TimeMLP

class ResBlock(nn.Module):
    """
    A simple residual block with time conditioning.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        # choose a valid group count that divides num_channels
        def choose_groups(ch, max_groups=8):
            for g in [8, 4, 2]:
                if ch % g == 0 and g <= max_groups:
                    return g
            return 1  # fallback

        g_in  = choose_groups(in_channels)
        g_out = choose_groups(out_channels)

        self.norm1 = nn.GroupNorm(g_in,  in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(g_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Map time embedding to (out_channels) to add as bias
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # time embedding -> (B, out_channels, 1, 1), add to h
        t_add = self.time_mlp(t_emb)
        h = h + t_add[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class UNetTime(nn.Module):
    """
    A small U-Net with time conditioning, outputting a vector field
    with same shape as input (B, 4, 128, 128).
    """

    def __init__(self, in_channels=4, base_channels=64, time_emb_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)

        # Downsampling path
        self.down1 = ResBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down4 = ResBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ResBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        # Upsampling path
        self.up4 = ResBlock(base_channels * 8 + base_channels * 8,
                            base_channels * 4,
                            time_emb_dim)
        self.up3 = ResBlock(base_channels * 4 + base_channels * 4,
                            base_channels * 2,
                            time_emb_dim)
        self.up2 = ResBlock(base_channels * 2 + base_channels * 2,
                            base_channels,
                            time_emb_dim)
        self.up1 = ResBlock(base_channels + base_channels,
                            base_channels,
                            time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Final output conv: map base_channels -> in_channels
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B,) or (B, 1) time in [0, 1]
        returns: velocity field (B, C, H, W)
        """
        # Build time embedding
        t_emb = self.time_embed(t)    # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # Down path
        h1 = self.down1(x, t_emb)                  # (B, base, 128,128)
        h2 = self.down2(self.pool(h1), t_emb)      # (B, 2*base, 64,64)
        h3 = self.down3(self.pool(h2), t_emb)      # (B, 4*base, 32,32)
        h4 = self.down4(self.pool(h3), t_emb)      # (B, 8*base,16,16)

        # Bottleneck
        mid = self.mid(self.pool(h4), t_emb)       # (B, 8*base, 8,8)

        # Up path
        u4 = self.upsample(mid)                    # (B, 8*base, 16,16)
        u4 = torch.cat([u4, h4], dim=1)            # concat skip
        u4 = self.up4(u4, t_emb)                   # (B, 4*base,16,16)

        u3 = self.upsample(u4)                    # (B, 4*base, 32,32)
        u3 = torch.cat([u3, h3], dim=1)            # concat skip
        u3 = self.up3(u3, t_emb)                   # (B, 2*base,32,32)

        u2 = self.upsample(u3)                     # (B,2*base,64,64)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up2(u2, t_emb)                   # (B, base,64,64)

        u1 = self.upsample(u2)                     # (B,base,128,128)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1(u1, t_emb)                   # (B, base,128,128)

        out = self.out_conv(u1)                    # (B, in_channels, 128,128)
        return out