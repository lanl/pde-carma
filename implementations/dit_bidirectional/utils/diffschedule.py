import torch


class DiffusionSchedule:
    """
    Linear DDPM schedule used for training noise and DDIM sampling.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        device=None,
        dtype=torch.float32,
    ):
        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps,
            device=device,
            dtype=dtype,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def normalized_time(self, t_idx):
        return t_idx.to(dtype=self.betas.dtype) / (self.num_timesteps - 1)

    def q_sample(self, x0, t_idx, noise):
        alpha_bar = self.alpha_bars[t_idx][:, None, None, None]

        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def ddim_timesteps(self, samp_diff_steps):

        ddim_timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            steps=samp_diff_steps,
            device=self.betas.device,
        ).round().long()

        return ddim_timesteps
