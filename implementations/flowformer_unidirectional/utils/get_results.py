import os
import numpy as np
import torch

from utils.visualize_flow_results import (
    visualize_real_vs_generated,
    visualize_full_trajectory,
)


class TestRolloutVisualizer:
    def __init__(
        self,
        model,
        vids_test,
        num_ode_steps,
        device,
        in_channels,
        save_results,
        max_T_rollout=20,
        num_sampling_frame=5,
        num_traj_rollout=1,
        num_eval=100,
        batch_size=4,
        run=0,
        extra_note="",
        direction="forward",
    ):
        self.model = model
        self.vids_test = vids_test.float()
        self.num_ode_steps = num_ode_steps
        self.device = device
        self.in_channels = in_channels
        self.save_results = save_results

        self.max_T_rollout = max_T_rollout
        self.num_sampling_frame = num_sampling_frame
        self.num_traj_rollout = num_traj_rollout
        self.num_eval = num_eval
        self.batch_size = batch_size
        self.run = run
        self.extra_note = extra_note
        self.direction = direction

        if self.direction not in ["forward", "backward"]:
            raise ValueError(
                f"Unsupported direction: {self.direction}. "
                "Expected 'forward' or 'backward'."
            )

    def evaluate_next_step(self):
        """
        Teacher-forced one-step prediction over flattened test pairs.

        Forward:
            true X_k     -> predicted X_{k+1}

        Backward:
            true X_k     -> predicted X_{k-1}

        Note:
            If vids_test has already been time-reversed for backward models,
            then the same indexing still works:
                vids_test[:, :-1] -> vids_test[:, 1:]
            corresponds to original:
                X_T -> X_{T-1}
        """

        N, T, C, H, W = self.vids_test.shape

        x_cond_all = self.vids_test[:, :-1].reshape(-1, C, H, W)
        x_true_all = self.vids_test[:, 1:].reshape(-1, C, H, W)

        total_pairs = x_cond_all.shape[0]
        num_eval = min(self.num_eval, total_pairs)

        print(f"Direction: {self.direction}")
        print(f"Total one-step pairs: {total_pairs}")
        print(f"Evaluating one-step prediction on {num_eval} pairs")

        x_pred_list = []
        x_true_list = []
        mse_list = []

        seen = 0

        for start in range(0, num_eval, self.batch_size):
            end = min(start + self.batch_size, num_eval)

            x_cond = x_cond_all[start:end]
            x_true = x_true_all[start:end]

            path = self.model.sample(
                x_cond=x_cond,
                num_ode_steps=self.num_ode_steps,
            )

            x_pred = path[:, -1]

            x_pred = x_pred.detach().cpu()
            x_true = x_true.detach().cpu()

            mse_per_sample = ((x_pred - x_true) ** 2).mean(dim=(1, 2, 3))

            x_pred_list.append(x_pred)
            x_true_list.append(x_true)
            mse_list.append(mse_per_sample)

            seen += x_pred.shape[0]

            print(
                f"NSP processed {seen}/{num_eval} | "
                f"Batch MSE: {mse_per_sample.mean().item():.6f}"
            )

        x_pred_all = torch.cat(x_pred_list, dim=0)
        x_true_all = torch.cat(x_true_list, dim=0)
        mse_all = torch.cat(mse_list, dim=0)

        save_path = os.path.join(
            self.save_results,
            f"nsp_results_{self.extra_note}_run{self.run}.pt",
        )

        torch.save(
            {
                "x_pred": x_pred_all,
                "x_true": x_true_all,
                "mse_per_sample": mse_all,
                "mean_mse": mse_all.mean(),
                "direction": self.direction,
            },
            save_path,
        )

        print(f"Saved NSP results to: {save_path}")
        print(f"Mean NSP MSE (Flow): {mse_all.mean().item():.6f}")

        return x_true_all, x_pred_all, mse_all

    def sample_single_step(self, frame_idx):
        """
        Visualize one-step prediction for a fixed frame index.

        If forward:
            X_frame_idx -> X_{frame_idx+1}

        If backward and vids_test is time-reversed:
            vids_test[:, frame_idx] -> vids_test[:, frame_idx+1]
            corresponds to original:
            X_T-frame_idx -> X_T-frame_idx-1
        """

        num_show = min(self.num_sampling_frame, self.vids_test.shape[0])
        N, T, C, H, W = self.vids_test.shape

        assert frame_idx >= 0, "frame_idx must be non-negative"
        assert frame_idx + 1 < T, "frame_idx too close to end for one-step viz"

        cond_idx = frame_idx
        target_idx = frame_idx + 1

        cond_batch = self.vids_test[:num_show, cond_idx]
        real_target = self.vids_test[:num_show, target_idx]

        path = self.model.sample(
            x_cond=cond_batch,
            num_ode_steps=self.num_ode_steps,
        )

        pred_target = path[:, -1]

        pred_target = pred_target.detach().cpu()
        real_target = real_target.detach().cpu()

        print("Single-step generated samples shape:", pred_target.shape)

        for ch in range(self.in_channels):
            visualize_real_vs_generated(
                real_target,
                pred_target,
                channel=ch,
                num_to_show=num_show,
                saveloc=self.save_results,
                frame_no=frame_idx,
                extra_note=f"{self.extra_note}_run{self.run}",
            )

        mse_per_sample = ((pred_target - real_target) ** 2).mean(dim=(1, 2, 3))

        return real_target, pred_target, mse_per_sample

    def rollout_and_visualize(self):
        """
        Autoregressive rollout.

        Forward:
            X_0 -> X_1 -> ... -> X_T

        Backward:
            If vids_test is time-reversed, this code still starts from
            vids_test[:, 0], which corresponds to original X_T.
            Then it rolls toward original X_0.
        """

        vids_subset = self.vids_test[:self.num_traj_rollout]

        N, T, C, H, W = vids_subset.shape
        T_roll = min(T, self.max_T_rollout)

        if T_roll < 2:
            raise ValueError(f"T_roll must be at least 2 for rollout, got {T_roll}")

        real_traj = vids_subset[:, :T_roll].detach().cpu()
        gen_traj = torch.zeros_like(real_traj)

        # For forward test data, this is original X_0.
        # For backward test data, this is original X_T because data was time-reversed.
        gen_traj[:, 0] = real_traj[:, 0]
        cond = real_traj[:, 0]

        for t_idx in range(1, T_roll):
            path = self.model.sample(
                x_cond=cond,
                num_ode_steps=self.num_ode_steps,
            )

            x_next = path[:, -1]
            x_next = x_next.detach().cpu()

            gen_traj[:, t_idx] = x_next
            cond = x_next

            print(f"Rollout generated step {t_idx}/{T_roll - 1}")

        mse_rollout = ((real_traj - gen_traj) ** 2).mean(dim=(2, 3, 4))

        for ch in range(self.in_channels):
            for seq_idx in range(N):
                visualize_full_trajectory(
                    real_traj,
                    gen_traj,
                    seq_idx=seq_idx,
                    channel=ch,
                    max_time_steps=T_roll,
                    saveloc=self.save_results,
                    extra_note=f"{self.extra_note}_run{self.run}",
                )

        mse_save_path = os.path.join(
            self.save_results,
            f"mse_rollout_{self.extra_note}_run{self.run}.txt",
        )

        np.savetxt(mse_save_path, mse_rollout.numpy())

        print(f"Saved rollout MSE to: {mse_save_path}")

        return real_traj, gen_traj, mse_rollout

    def run_all(self, frame_idx):
        real_nsp, pred_nsp, mse_nsp = self.evaluate_next_step()

        real_single, pred_single, mse_single = self.sample_single_step(frame_idx)

        real_rollout, gen_rollout, mse_rollout = self.rollout_and_visualize()

        return {
            "real_nsp": real_nsp,
            "pred_nsp": pred_nsp,
            "mse_nsp": mse_nsp,
            "real_single": real_single,
            "pred_single": pred_single,
            "mse_single": mse_single,
            "real_rollout": real_rollout,
            "gen_rollout": gen_rollout,
            "mse_rollout": mse_rollout,
        }