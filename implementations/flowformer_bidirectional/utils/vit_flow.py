import torch
import torch.nn as nn

from utils.time_embedding import SinusoidalTimeEmbedding, TimeMLP
from utils.pos_enc import LearnablePositionalEncoding
from utils.transformer_main import TransformerBackbone
from utils.patchify import PatchEmbed, PatchDecoder


def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2, f"Expected int or length-2 tuple, got {x}"
    return tuple(x)


class ViTTime(nn.Module):
    """
    Vision Transformer backbone for conditional flow matching.

    Forward control:
        c = [1, 0] -> forward prediction:  x_t     -> x_{t+1}
        c = [0, 1] -> backward prediction: x_{t+1} -> x_t

    Input:
        x: (B, 2C, H, W) = [x_cond, x_flow_state]
        t: (B,) flow time in [0, 1]
        c: (B, 2) direction/control condition

    Output:
        out: (B, C, H, W) = velocity field for target frame
    """

    def __init__(
        self,
        in_channels=5,
        embed_dim=128,
        time_emb_dim=512,
        img_size=128,
        patch_size=8,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        img_h, img_w = img_size
        patch_h, patch_w = patch_size

        self.physical_channels = in_channels
        self.in_channels = in_channels * 2
        self.out_channels = in_channels

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_h // patch_h, img_w // patch_w)
        self.embed_dim = embed_dim

        # Flow-time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, embed_dim)

        # Direction/control embedding:
        # c=[1,0] forward, c=[0,1] backward
        self.control_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Patchify 2C input channels
        self.patch_embed = PatchEmbed(
            in_channels=self.in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.pos_encoding = LearnablePositionalEncoding(
            grid_size=self.grid_size,
            embed_dim=embed_dim,
        )

        self.transformer = TransformerBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Decode to C output channels
        self.patch_decoder = PatchDecoder(
            out_channels=self.out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

    def _default_forward_control(self, B, device, dtype):
        c = torch.zeros(B, 2, device=device, dtype=dtype)
        c[:, 0] = 1.0
        return c

    def forward(self, x, t, c=None):
        """
        x: (B, 2C, H, W)
        t: (B,) or (B, 1)
        c: (B, 2), (2,), or None

        If c is None, defaults to forward mode c=[1,0].
        """

        B, C, H, W = x.shape

        assert C == self.in_channels, (
            f"Expected {self.in_channels} channels, got {C}"
        )

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        t = t.to(device=x.device, dtype=x.dtype)

        if t.ndim == 0:
            t = t.repeat(B)

        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)

        # Flow-time token
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        t_token = self.time_proj(t_emb)

        # Control token
        if c is None:
            c = self._default_forward_control(B, x.device, x.dtype)
        else:
            c = c.to(device=x.device, dtype=x.dtype)

            if c.ndim == 1:
                assert c.numel() == 2, f"Expected c shape (2,), got {c.shape}"
                c = c.unsqueeze(0).expand(B, -1)

            assert c.shape == (B, 2), f"Expected c shape ({B}, 2), got {c.shape}"

        c_token = self.control_mlp(c)

        # Patchify
        tokens, H_p, W_p = self.patch_embed(x)

        # Positional + time + control conditioning
        tokens = self.pos_encoding(tokens, H_p, W_p)
        tokens = tokens + t_token[:, None, :] + c_token[:, None, :]

        # Transformer
        tokens = self.transformer(tokens)

        # Decode
        out = self.patch_decoder(tokens, H_p, W_p)

        return out