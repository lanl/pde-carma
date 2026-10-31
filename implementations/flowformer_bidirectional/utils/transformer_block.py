import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
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