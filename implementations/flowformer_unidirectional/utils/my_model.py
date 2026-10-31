import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.inference_mode()
    def predict_velocity(self, x_cond, x_t, t):
        """
        x_cond: [B, C, H, W]
        x_t:    [B, C, H, W]
        t:      [B]
        """

        x_cond = x_cond.to(self.device)
        x_t = x_t.to(self.device)
        t = t.to(device=self.device, dtype=x_cond.dtype)

        if x_cond.shape != x_t.shape:
            raise ValueError(
                f"x_cond and x_t must have same shape. "
                f"Got x_cond={x_cond.shape}, x_t={x_t.shape}"
            )

        x_in = torch.cat([x_cond, x_t], dim=1)

        v = self.model(x_in, t)

        return v

    @torch.inference_mode()
    def sample(self, x_cond, num_ode_steps=None, return_traj=False):
        
        self.eval()

        x_cond = x_cond.to(self.device)

        B, C, H, W = x_cond.shape

        if num_ode_steps is None:
            num_ode_steps = self.config.get("num_ode_steps", 500)

        # Start from noise
        x_t = torch.randn_like(x_cond)

        dt = 1.0 / num_ode_steps

        traj = []

        for i in range(num_ode_steps):

            # pseudo-time t in [0, 1]
            t_val = i / num_ode_steps

            # broadcast t to shape (B,)
            t = torch.full(
                (B,),
                fill_value=t_val,
                device=self.device,
                dtype=x_cond.dtype,
            )

            # Predict velocity
            v = self.predict_velocity(x_cond, x_t, t)

            # Euler update
            x_t = x_t + dt * v

            # full path
            traj.append(x_t.detach().cpu())

        path = torch.stack(traj, dim=1)

        return path

    @torch.inference_mode()
    def predict(self, x_cond, num_ode_steps=None):

        path = self.sample(
            x_cond=x_cond,
            num_ode_steps=num_ode_steps,
        )

        x_pred = path[:, -1]  # only final state of the path 

        return x_pred
