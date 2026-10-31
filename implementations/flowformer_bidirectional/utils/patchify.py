import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2, f"Expected int or length-2 tuple, got {x}"
    return tuple(x)


class PatchEmbed(nn.Module):
    """
    Converts image/field into patch tokens.

    Input:
        x: (B, C, H, W)

    Output:
        tokens: (B, N, D)
        H_p, W_p: patch grid size
    """

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        h = self.proj(x)                       # (B, D, H/P_h, W/P_w)
        H_p, W_p = h.shape[-2], h.shape[-1]

        tokens = h.flatten(2).transpose(1, 2)  # (B, N, D)

        return tokens, H_p, W_p


class PatchDecoder(nn.Module):
    """
    Converts patch tokens back to full-resolution image/field.

    Supports non-square patch sizes, e.g. patch_size=(8, 16).

    Input:
        tokens: (B, N, D)

    Output:
        out: (B, C, H, W)
    """

    def __init__(self, out_channels, embed_dim, patch_size):
        super().__init__()

        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        patch_h, patch_w = self.patch_size

        self.to_patch_pixels = nn.Conv2d(
            embed_dim,
            out_channels * patch_h * patch_w,
            kernel_size=1,
        )

    def forward(self, tokens, H_p, W_p):
        B, N, D = tokens.shape
        patch_h, patch_w = self.patch_size

        assert N == H_p * W_p, \
            f"Token count N={N} does not match H_p*W_p={H_p * W_p}"

        h = tokens.transpose(1, 2).reshape(B, D, H_p, W_p)
        h = self.to_patch_pixels(h)  # (B, C*pH*pW, H_p, W_p)

        h = h.view(
            B,
            self.out_channels,
            patch_h,
            patch_w,
            H_p,
            W_p,
        )

        # (B, C, pH, pW, H_p, W_p)
        # -> (B, C, H_p, pH, W_p, pW)
        h = h.permute(0, 1, 4, 2, 5, 3).contiguous()

        out = h.view(
            B,
            self.out_channels,
            H_p * patch_h,
            W_p * patch_w,
        )

        return out