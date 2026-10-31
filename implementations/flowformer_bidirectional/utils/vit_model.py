import torch.nn as nn
import torch
from utils.transformer_block import TransformerBlock

# -----------------------------
# 3) Simple ViT image->image
# -----------------------------
class SimpleViTImg2Img(nn.Module):
    def __init__(
        self,
        in_channels: int,        
        img_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.F = in_channels
        self.P = patch_size
        self.HW = img_size
        self.nH = img_size // patch_size
        self.nW = img_size // patch_size
        self.num_patches = self.nH * self.nW
        self.embed_dim = embed_dim

        # Patchify: (B,F,H,W) -> (B, D, nH, nW) -> flatten -> (B, Patches, D)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Head maps each token to its patch's raw pixels (F * P * P)
        self.head = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

        # (Optional) init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, F, H, W)
        B, F, H, W = x.shape
        assert F == self.F and H == self.HW and W == self.HW, "Input must match configured (F,H,W)"

        # Patchify
        tok = self.patch_embed(x)                              # (B, D, nH, nW)
        tok = tok.flatten(2).transpose(1, 2)                   # (B, Patches, D)
        tok = tok + self.pos_embed                             # add positional info

        # Transformer encoder
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)                                   # (B, Patches, D)

        # Predict patch pixels
        patch_pixels = self.head(tok)                          # (B, Patches, F*P*P)

        # Unpatchify back to image
        out = patch_pixels.view(B, self.nH, self.nW, self.F, self.P, self.P)  # (B,nH,nW,F,P,P)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()                       # (B,F,nH,P,nW,P)
        out = out.view(B, self.F, self.nH * self.P, self.nW * self.P)         # (B,F,H,W)
        return out