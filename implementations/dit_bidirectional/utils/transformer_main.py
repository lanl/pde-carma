import torch.nn as nn

from utils.transformer_block import DiTBlock, TransformerBlock, modulate


class TransformerBackbone(nn.Module):
    """Legacy stack of standard Transformer blocks."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)


class DiTBackbone(nn.Module):
    """Transformer stack with DiT-style adaLN-Zero conditioning in every block."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=dropout,
            )
            for _ in range(depth)
        ])

        # Match the DiT final layer idea: condition the final normalized tokens too.
        self.norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def forward(self, tokens, cond):
        for block in self.blocks:
            tokens = block(tokens, cond)

        shift, scale = self.final_modulation(cond).chunk(2, dim=1)
        return modulate(self.norm(tokens), shift, scale)
