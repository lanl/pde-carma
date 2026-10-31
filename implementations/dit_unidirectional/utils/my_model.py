import torch
import torch.nn as nn

from utils.diffschedule import DiffusionSchedule
from utils.vit_diffusion import ViTTime
from utils.data_config import data_config
from utils.model_config import model_config


class DiTInference(nn.Module):
    def __init__(self, model, config, device="cpu"):
        super().__init__()

        self.model = model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(cls, checkpoint_path, img_size, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)

        config = ckpt["args"]

        print("===== Pretrained DiT config =====")
        for k in sorted(config.keys()):
            print(f"{k}: {config[k]}")
        print("===============================")

        ds = config["ds"]
        ms = config["ms"]

        data_configs = data_config()
        model_configs = model_config()

        channels = data_configs[ds]["channels"]

        model_dims = model_configs[ms]["model_dims"]
        num_heads = model_configs[ms]["num_heads"]
        depth = model_configs[ms]["depth"]

        patch_size = tuple(config["patch_size"])
        dit_arch = config.get("dit_arch", "legacy")

        model = ViTTime(
            in_channels=channels,
            time_emb_dim=config["time_emb_dim"],
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=model_dims,
            depth=depth,
            num_heads=num_heads,
            dit_arch=dit_arch,
        )

        state_dict = ckpt["model_state_dict"]

        if len(state_dict) > 0 and next(iter(state_dict)).startswith("module."):
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        inference_model = cls(
            model=model,
            config=config,
            device=device,
        )

        inference_model.to(device)
        inference_model.eval()

        num_params = sum(p.numel() for p in inference_model.parameters())
        print(f"TOTAL PARAMS: {num_params / 1e6:.4f} M")

        return inference_model

    def _prepare_control(self, c, B, dtype):
        """
        c = [1,0] forward
        c = [0,1] backward
        """

        if c is None:
            c = torch.zeros(B, 2, device=self.device, dtype=dtype)
            c[:, 0] = 1.0
            return c

        if not torch.is_tensor(c):
            c = torch.tensor(c, device=self.device, dtype=dtype)
        else:
            c = c.to(device=self.device, dtype=dtype)

        if c.ndim == 1:
            assert c.numel() == 2, f"Expected c with 2 values, got {c.shape}"
            c = c.unsqueeze(0).expand(B, -1)

        assert c.shape == (B, 2), f"Expected c shape ({B}, 2), got {c.shape}"

        return c

    def _schedule(self, dtype):
        num_timesteps = self.config.get("num_diffusion_steps", 1000)

        return DiffusionSchedule(
            num_timesteps=num_timesteps,
            beta_start=self.config.get("beta_start", 1e-4),
            beta_end=self.config.get("beta_end", 2e-2),
            device=self.device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def predict_noise(self, x_cond, x_t, t, c=None):
        """
        Predict epsilon for the noisy target frame.

        x_cond: [B, C, H, W]
        x_t:    [B, C, H, W]
        t:      [B], normalized diffusion time in [0, 1]
        c:      [B, 2], [2], or None
        """

        x_cond = x_cond.to(self.device)
        x_t = x_t.to(self.device)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=x_cond.dtype)
        else:
            t = t.to(device=self.device, dtype=x_cond.dtype)

        if x_cond.shape != x_t.shape:
            raise ValueError(
                f"x_cond and x_t must have same shape. "
                f"Got x_cond={x_cond.shape}, x_t={x_t.shape}"
            )

        B = x_cond.shape[0]
        c = self._prepare_control(c, B, x_cond.dtype)

        x_in = torch.cat([x_cond, x_t], dim=1)

        noise = self.model(x_in, t, c)

        return noise

    @torch.inference_mode()
    def sample(self, x_cond, samp_diff_steps=None, num_ode_steps=None, c=None, ddim_eta=None, return_path=True,):
        """
        Generate one next state with DDIM reverse diffusion.

        samp_diff_steps means reverse diffusion sampling steps.
        """

        self.eval()

        x_cond = x_cond.to(self.device)

        B = x_cond.shape[0]
        c = self._prepare_control(c, B, x_cond.dtype)

        schedule = self._schedule(dtype=x_cond.dtype)

        if samp_diff_steps is None and num_ode_steps is not None:
            samp_diff_steps = num_ode_steps

        if samp_diff_steps is None:
            samp_diff_steps = self.config.get("samp_diff_steps", schedule.num_timesteps)

        if ddim_eta is None:
            ddim_eta = self.config.get("ddim_eta", 0.0)

        timesteps = schedule.ddim_timesteps(samp_diff_steps)
        x_t = torch.randn_like(x_cond)

        traj = [] if return_path else None

        for step_idx, timestep in enumerate(timesteps):
            is_last_step = step_idx == len(timesteps) - 1

            t_idx = torch.full(
                (B,),
                fill_value=int(timestep.item()),
                device=self.device,
                dtype=torch.long,
            )

            t = schedule.normalized_time(t_idx)

            noise_pred = self.predict_noise(
                x_cond=x_cond,
                x_t=x_t,
                t=t,
                c=c,
            )

            alpha_bar_t = schedule.alpha_bars[timestep]
            x0_pred = (
                x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_bar_t)

            if is_last_step:
                alpha_bar_prev = torch.ones((), device=self.device, dtype=x_t.dtype)
            else:
                alpha_bar_prev = schedule.alpha_bars[timesteps[step_idx + 1]]

            if ddim_eta > 0.0 and not is_last_step:
                sigma = ddim_eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_bar_prev)
                        / (1.0 - alpha_bar_t)
                        * (1.0 - alpha_bar_t / alpha_bar_prev),
                        min=0.0,
                    )
                )
                z = torch.randn_like(x_t)
            else:
                sigma = torch.zeros((), device=self.device, dtype=x_t.dtype)
                z = 0.0

            direction_scale = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)
            )

            x_t = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + direction_scale * noise_pred
                + sigma * z
            )

            if return_path:
                traj.append(x_t.detach().cpu())

        if return_path:
            return torch.stack(traj, dim=1)

        return x_t

    @torch.inference_mode()
    def predict(self, x_cond, samp_diff_steps=None, c=None):
        path = self.sample(
            x_cond=x_cond,
            samp_diff_steps=samp_diff_steps,
            c=c,
        )

        x_pred = path[:, -1]

        return x_pred
