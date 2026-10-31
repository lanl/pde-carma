import torch
import torch.nn as nn
from utils.transformer_block import TransformerBlock

class TransformerBackbone(nn.Module):
    """
    Stack of Transformer blocks.
    """

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

        tokens = self.norm(tokens)

        return tokens