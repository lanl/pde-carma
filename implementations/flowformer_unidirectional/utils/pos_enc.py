import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2, f"Expected int or length-2 tuple, got {x}"
    return tuple(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding for ViT patch tokens.
    Supports rectangular grids.
    """

    def __init__(self, grid_size, embed_dim):
        super().__init__()

        self.grid_size = to_2tuple(grid_size)
        self.embed_dim = embed_dim

        grid_h, grid_w = self.grid_size
        self.num_patches = grid_h * grid_w

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens, H_p, W_p):
        """
        tokens: (B, N, D)
        """
        grid_h, grid_w = self.grid_size

        # Reshape to (1, grid_h, grid_w, D)
        pos = self.pos_embed.reshape(1, grid_h, grid_w, self.embed_dim)

        # Permute to (1, D, grid_h, grid_w) for interpolation
        pos = pos.permute(0, 3, 1, 2)  # (1, D, grid_h, grid_w)

        # Interpolate to (1, D, H_p, W_p)
        pos = F.interpolate(
            pos,
            size=(H_p, W_p),
            mode="bicubic",
            align_corners=False,
        )
        # Reshape back to (1, N, D)
        pos = pos.permute(0, 2, 3, 1)
        pos = pos.reshape(1, H_p * W_p, self.embed_dim)

        return tokens + pos