import torch
import torch.nn as nn


def modulate(x, shift, scale):
    """Apply adaptive LayerNorm shift/scale to token features."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Legacy pre-norm Transformer block used by the original BiDiT."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """
        x: (B, N, D)
        """
        h = self.norm1(x)
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.mlp(h)

        return x


class DiTBlock(nn.Module):
    """DiT-style Transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()

        # adaLN supplies the affine parameters, so LayerNorm has no affine terms.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

        # One conditioning vector generates shift/scale/gate for attention and MLP.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # adaLN-Zero: each residual branch starts gated off.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        """
        x:    (B, N, D)
        cond: (B, D), combined diffusion-time + direction embedding
        """
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(cond).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * attn_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x
