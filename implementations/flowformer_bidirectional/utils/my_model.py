import torch
import torch.nn as nn

from utils.vit_flow import ViTTime
from utils.data_config import data_config
from utils.model_config import model_config


class FlowFormerInference(nn.Module):
    def __init__(self, model, config, device="cpu"):
        super().__init__()

        self.model = model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(cls, checkpoint_path, img_size, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)

        config = ckpt["args"]

        print("===== Pretrained FlowFormer config =====")
        for k in sorted(config.keys()):
            print(f"{k}: {config[k]}")
        print("========================================")

        ds = config["ds"]
        ms = config["ms"]

        data_configs = data_config()
        model_configs = model_config()

        channels = data_configs[ds]["channels"]

        model_dims = model_configs[ms]["model_dims"]
        num_heads = model_configs[ms]["num_heads"]
        depth = model_configs[ms]["depth"]

        patch_size = tuple(config["patch_size"])

        model = ViTTime(
            in_channels=channels,
            time_emb_dim=config["time_emb_dim"],
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=model_dims,
            depth=depth,
            num_heads=num_heads,
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

    def _default_forward_control(self, B, dtype):
        c = torch.zeros(B, 2, device=self.device, dtype=dtype)
        c[:, 0] = 1.0
        return c

    def _prepare_control(self, c, B, dtype):
        """
        c = [1,0] forward
        c = [0,1] backward
        """

        if c is None:
            return self._default_forward_control(B, dtype)

        if not torch.is_tensor(c):
            c = torch.tensor(c, device=self.device, dtype=dtype)
        else:
            c = c.to(device=self.device, dtype=dtype)

        if c.ndim == 1:
            assert c.numel() == 2, f"Expected c with 2 values, got {c.shape}"
            c = c.unsqueeze(0).expand(B, -1)

        assert c.shape == (B, 2), f"Expected c shape ({B}, 2), got {c.shape}"

        return c

    @torch.inference_mode()
    def predict_velocity(self, x_cond, x_t, t, c=None):
        """
        x_cond: [B, C, H, W]
        x_t:    [B, C, H, W]
        t:      [B]
        c:      [B, 2], [2], or None
                [1,0] forward
                [0,1] backward
        """

        x_cond = x_cond.to(self.device)
        x_t = x_t.to(self.device)
        t = t.to(device=self.device, dtype=x_cond.dtype)

        if x_cond.shape != x_t.shape:
            raise ValueError(
                f"x_cond and x_t must have same shape. "
                f"Got x_cond={x_cond.shape}, x_t={x_t.shape}"
            )

        B = x_cond.shape[0]
        c = self._prepare_control(c, B, x_cond.dtype)

        x_in = torch.cat([x_cond, x_t], dim=1)

        v = self.model(x_in, t, c)

        return v

    @torch.inference_mode()
    def sample(self, x_cond, num_ode_steps=None, return_traj=False, c=None):
        """
        Generate one next state by integrating the learned flow.

        x_cond: [B, C, H, W]
        c:
            [1,0] forward mode
            [0,1] backward mode

        return_traj is kept for compatibility. This function currently
        returns the full ODE path either way, matching the original behavior.
        """

        self.eval()

        x_cond = x_cond.to(self.device)

        B, C, H, W = x_cond.shape
        c = self._prepare_control(c, B, x_cond.dtype)

        if num_ode_steps is None:
            num_ode_steps = self.config.get("num_ode_steps", 500)

        x_t = torch.randn_like(x_cond)

        dt = 1.0 / num_ode_steps

        traj = []

        for i in range(num_ode_steps):
            t_val = i / num_ode_steps

            t = torch.full(
                (B,),
                fill_value=t_val,
                device=self.device,
                dtype=x_cond.dtype,
            )

            v = self.predict_velocity(
                x_cond=x_cond,
                x_t=x_t,
                t=t,
                c=c,
            )

            x_t = x_t + dt * v

            traj.append(x_t.detach().cpu())

        path = torch.stack(traj, dim=1)

        return path

    @torch.inference_mode()
    def predict(self, x_cond, num_ode_steps=None, c=None):
        path = self.sample(
            x_cond=x_cond,
            num_ode_steps=num_ode_steps,
            c=c,
        )

        x_pred = path[:, -1]

        return x_pred