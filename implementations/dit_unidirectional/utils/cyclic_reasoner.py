import torch
from utils.diffusion_trainer import sample_from_model

class CyclicReasoner:
    def __init__(self, model_F, model_G, device):
        """
        Run multiple forward trajectories from x_start_F, then run a backward
        trajectory from each forward endpoint and compute cycle-consistency.

        Returns:
        fwd_traj:     (n_rand_samples, H_F+1, C, H, W)
        bwd_traj:     (n_rand_samples, H_R+1, C, H, W)
        cycle_errors: (n_rand_samples,) MSE between x_end_R (reconstructed x_0) and x_start_F
        forward_idx:  (n_rand_samples,) index of the forward trajectory
        branch_idx:   (n_rand_samples,) always 0 (one backward branch per forward traj)
        """
        
        self.model_F = model_F
        self.model_G = model_G
        self.device = device

    @torch.no_grad()
    def rollout_model_batch(self, model, x0_batch, horizon,
                            num_steps_per_frame):
        """
        Roll out `model` starting from x0_batch for `horizon` steps.

        Args:
            model: neural net
            x0_batch: (B, C, H, W) or (C, H, W)
            horizon: int, number of steps
            num_steps_per_frame: ODE steps per frame
            device: torch.device

        Returns:
            traj: (B, horizon+1, C, H, W) on CPU
        """
        model.eval()

        if x0_batch.ndim == 3:
            x0_batch = x0_batch.unsqueeze(0)  # (1, C, H, W)

        cond = x0_batch.to(self.device)            # (B, C, H, W)
        B, C, H, W = cond.shape

        traj = torch.zeros(B, horizon + 1, C, H, W, device=self.device)
        traj[:, 0] = cond

        for t in range(1, horizon + 1):
            x_next = sample_from_model(model, cond, num_steps=num_steps_per_frame,
                                       device=self.device)  # expected shape: (B, C, H, W)
            traj[:, t] = x_next
            cond = x_next

        return traj


    @torch.no_grad()
    def sample_forward_backward_cycles(self,
        x_start_F,                  # (C, H, W) tensor, ground truth frame at t_start
        H_F,                        # forward horizon (number of steps)
        H_R,                        # backward horizon (number of steps)
        n_rand_samples,             # number of forward/backward trajectories
        num_steps_F,                # ODE steps for forward sampling
        num_steps_R,                # ODE steps for backward sampling
    ):
        C, H, W = x_start_F.shape

        # ----- Forward sampling with F -----
        # Same x_start_F repeated n_rand_samples times
        x0_F_batch = x_start_F.unsqueeze(0).expand(n_rand_samples, -1, -1, -1)
        # Shape: (n_rand_samples, H_F+1, C, H, W)
        fwd_traj = self.rollout_model_batch(self.model_F, x0_F_batch, horizon=H_F,
                                            num_steps_per_frame=num_steps_F)

        # Endpoints for each forward trajectory: (n_rand_samples, C, H, W)
        x_end_F = fwd_traj[:, -1]

        # ----- Backward sampling with G -----
        # One backward trajectory per forward endpoint
        x0_R_batch = x_end_F  # (n_rand_samples, C, H, W)

        # Shape: (n_rand_samples, H_R+1, C, H, W)
        bwd_traj = self.rollout_model_batch(self.model_G, x0_R_batch, horizon=H_R,
                                            num_steps_per_frame=num_steps_R)

        # Final frames after backward rollout, interpreted as reconstructed x_0
        x_end_R = bwd_traj[:, -1]   # (n_rand_samples, C, H, W)

        # ----- Cycle-consistency error: compare reconstructed x_0 to true x_start_F -----
        x_start_batch = x_start_F.unsqueeze(0).expand(n_rand_samples, -1, -1, -1).to(self.device)  # (n_rand_samples, C, H, W)
        cycle_errors = ((x_end_R - x_start_batch) ** 2).mean(dim=(1, 2, 3))  # (n_rand_samples,)
        
        # Indices
        forward_idx = torch.arange(n_rand_samples, device=self.device)

        # Since we have only one backward branch per forward traj
        branch_idx = torch.zeros(n_rand_samples, dtype=torch.long, device=self.device)

        return fwd_traj, bwd_traj, cycle_errors, forward_idx, branch_idx