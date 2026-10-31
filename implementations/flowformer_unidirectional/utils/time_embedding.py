import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding:
    t in [0, 1] -> embedding vector (B, time_dim).
    """

    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, t):
        """
        t: (B,) or (B, 1) in [0,1]
        Returns: (B, time_dim)
        """
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        half_dim = self.time_dim // 2
        # make frequencies
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                device=t.device,
            )
        )
        # outer product: (B, half_dim)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class TimeMLP(nn.Module):
    """
    MLP to map sinusoidal time embedding to desired hidden dim. 
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
