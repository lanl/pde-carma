import os
import time
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
        ddim_eta,
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
        cyclic_runs=None,
    ):
        self.model = model
        self.vids_test = vids_test.float()
        self.num_ode_steps = num_ode_steps
        self.ddim_eta = ddim_eta
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
        self.cyclic_runs = cyclic_runs if cyclic_runs is not None else [1]

        self.c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.c_backward = torch.tensor([0.0, 1.0], dtype=torch.float32)

    def evaluate_next_step(self):
        N, T, C, H, W = self.vids_test.shape

        x_cond_all = self.vids_test[:, :-1].reshape(-1, C, H, W)
        x_true_all = self.vids_test[:, 1:].reshape(-1, C, H, W)

        total_pairs = x_cond_all.shape[0]
        num_eval = min(self.num_eval, total_pairs)

        print(f"Total forward next-step pairs: {total_pairs}")
        print(f"Evaluating forward next-step prediction on {num_eval} pairs")

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
                ddim_eta=self.ddim_eta,
                c=self.c_forward,
            )

            x_pred = path[:, -1].detach().cpu()
            x_true = x_true.detach().cpu()

            mse_per_sample = ((x_pred - x_true) ** 2).mean(dim=(1, 2, 3))

            x_pred_list.append(x_pred)
            x_true_list.append(x_true)
            mse_list.append(mse_per_sample)

            seen += x_pred.shape[0]

            print(
                f"Forward NSP processed {seen}/{num_eval} | "
                f"Batch MSE: {mse_per_sample.mean().item():.6f}"
            )

        x_pred_all = torch.cat(x_pred_list, dim=0)
        x_true_all = torch.cat(x_true_list, dim=0)
        mse_all = torch.cat(mse_list, dim=0)

        save_path = os.path.join(
            self.save_results,
            f"nsp_forward_results_{self.extra_note}_run{self.run}.pt",
        )

        torch.save(
            {
                "x_pred": x_pred_all,
                "x_true": x_true_all,
                "mse_per_sample": mse_all,
                "mean_mse": mse_all.mean(),
                "control": self.c_forward,
            },
            save_path,
        )

        print(f"Saved forward NSP results to: {save_path}")
        print(f"Mean forward NSP MSE (DiT): {mse_all.mean().item():.6f}")

        return x_true_all, x_pred_all, mse_all

    def evaluate_previous_step(self):
        N, T, C, H, W = self.vids_test.shape

        x_cond_all = self.vids_test[:, 1:].reshape(-1, C, H, W)
        x_true_all = self.vids_test[:, :-1].reshape(-1, C, H, W)

        total_pairs = x_cond_all.shape[0]
        num_eval = min(self.num_eval, total_pairs)

        print(f"Total backward previous-step pairs: {total_pairs}")
        print(f"Evaluating backward previous-step prediction on {num_eval} pairs")

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
                ddim_eta=self.ddim_eta,
                c=self.c_backward,
            )

            x_pred = path[:, -1].detach().cpu()
            x_true = x_true.detach().cpu()

            mse_per_sample = ((x_pred - x_true) ** 2).mean(dim=(1, 2, 3))

            x_pred_list.append(x_pred)
            x_true_list.append(x_true)
            mse_list.append(mse_per_sample)

            seen += x_pred.shape[0]

            print(
                f"Backward NSP processed {seen}/{num_eval} | "
                f"Batch MSE: {mse_per_sample.mean().item():.6f}"
            )

        x_pred_all = torch.cat(x_pred_list, dim=0)
        x_true_all = torch.cat(x_true_list, dim=0)
        mse_all = torch.cat(mse_list, dim=0)

        save_path = os.path.join(
            self.save_results,
            f"nsp_backward_results_{self.extra_note}_run{self.run}.pt",
        )

        torch.save(
            {
                "x_pred": x_pred_all,
                "x_true": x_true_all,
                "mse_per_sample": mse_all,
                "mean_mse": mse_all.mean(),
                "control": self.c_backward,
            },
            save_path,
        )

        print(f"Saved backward NSP results to: {save_path}")
        print(f"Mean backward NSP MSE (DiT): {mse_all.mean().item():.6f}")

        return x_true_all, x_pred_all, mse_all

    def sample_single_step(self, frame_idx):
        num_show = min(self.num_sampling_frame, self.vids_test.shape[0])
        N, T, C, H, W = self.vids_test.shape

        assert frame_idx >= 0, "frame_idx must be non-negative"
        assert frame_idx + 1 < T, "frame_idx too close to end for forward next-step viz"

        cond_batch = self.vids_test[:num_show, frame_idx]
        real_target = self.vids_test[:num_show, frame_idx + 1]

        path = self.model.sample(
            x_cond=cond_batch,
            num_ode_steps=self.num_ode_steps,
            ddim_eta=self.ddim_eta,
            c=self.c_forward,
        )

        pred_target = path[:, -1].detach().cpu()
        real_target = real_target.detach().cpu()

        print("Forward single-step generated samples shape:", pred_target.shape)

        for ch in range(self.in_channels):
            visualize_real_vs_generated(
                real_target,
                pred_target,
                channel=ch,
                num_to_show=num_show,
                saveloc=self.save_results,
                frame_no=frame_idx,
                extra_note=f"{self.extra_note}_forward_run{self.run}",
            )

        mse_per_sample = ((pred_target - real_target) ** 2).mean(dim=(1, 2, 3))

        return real_target, pred_target, mse_per_sample

    def sample_single_step_backward(self, frame_idx):
        num_show = min(self.num_sampling_frame, self.vids_test.shape[0])
        N, T, C, H, W = self.vids_test.shape

        assert frame_idx >= 1, "frame_idx must be at least 1 for backward previous-step viz"
        assert frame_idx < T, "frame_idx out of range"

        cond_batch = self.vids_test[:num_show, frame_idx]
        real_target = self.vids_test[:num_show, frame_idx - 1]

        path = self.model.sample(
            x_cond=cond_batch,
            num_ode_steps=self.num_ode_steps,
            ddim_eta=self.ddim_eta,
            c=self.c_backward,
        )

        pred_target = path[:, -1].detach().cpu()
        real_target = real_target.detach().cpu()

        print("Backward single-step generated samples shape:", pred_target.shape)

        for ch in range(self.in_channels):
            visualize_real_vs_generated(
                real_target,
                pred_target,
                channel=ch,
                num_to_show=num_show,
                saveloc=self.save_results,
                frame_no=frame_idx,
                extra_note=f"{self.extra_note}_backward_run{self.run}",
            )

        mse_per_sample = ((pred_target - real_target) ** 2).mean(dim=(1, 2, 3))

        return real_target, pred_target, mse_per_sample

    def rollout_and_visualize(self):
        vids_subset = self.vids_test[: self.num_traj_rollout]

        N, T, C, H, W = vids_subset.shape
        T_roll = min(T, self.max_T_rollout)

        if T_roll < 2:
            raise ValueError(f"T_roll must be at least 2 for rollout, got {T_roll}")

        real_traj = vids_subset[:, :T_roll].detach().cpu()
        gen_traj = torch.zeros_like(real_traj)

        gen_traj[:, 0] = real_traj[:, 0]
        cond = real_traj[:, 0]

        for t_idx in range(1, T_roll):
            path = self.model.sample(
                x_cond=cond,
                num_ode_steps=self.num_ode_steps,
                ddim_eta=self.ddim_eta,
                c=self.c_forward,
            )

            x_next = path[:, -1].detach().cpu()

            gen_traj[:, t_idx] = x_next
            cond = x_next

            print(f"Forward rollout generated step {t_idx}/{T_roll - 1}")

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
                    extra_note=f"{self.extra_note}_forward_run{self.run}",
                )

        mse_save_path = os.path.join(
            self.save_results,
            f"mse_forward_rollout_{self.extra_note}_run{self.run}.txt",
        )

        np.savetxt(mse_save_path, mse_rollout.numpy())

        print(f"Saved forward rollout MSE to: {mse_save_path}")

        return real_traj, gen_traj, mse_rollout

    def backward_rollout_and_visualize(self):
        vids_subset = self.vids_test[: self.num_traj_rollout]

        N, T, C, H, W = vids_subset.shape
        T_roll = min(T, self.max_T_rollout)

        if T_roll < 2:
            raise ValueError(f"T_roll must be at least 2 for backward rollout, got {T_roll}")

        real_forward = vids_subset[:, :T_roll].detach().cpu()
        real_backward = torch.flip(real_forward, dims=[1])

        gen_backward = torch.zeros_like(real_backward)

        gen_backward[:, 0] = real_backward[:, 0]
        cond = real_backward[:, 0]

        for t_idx in range(1, T_roll):
            path = self.model.sample(
                x_cond=cond,
                num_ode_steps=self.num_ode_steps,
                ddim_eta=self.ddim_eta,
                c=self.c_backward,
            )

            x_prev = path[:, -1].detach().cpu()

            gen_backward[:, t_idx] = x_prev
            cond = x_prev

            print(f"Backward rollout generated step {t_idx}/{T_roll - 1}")

        mse_backward = ((real_backward - gen_backward) ** 2).mean(dim=(2, 3, 4))

        for ch in range(self.in_channels):
            for seq_idx in range(N):
                visualize_full_trajectory(
                    real_backward,
                    gen_backward,
                    seq_idx=seq_idx,
                    channel=ch,
                    max_time_steps=T_roll,
                    saveloc=self.save_results,
                    extra_note=f"{self.extra_note}_backward_run{self.run}",
                )

        mse_save_path = os.path.join(
            self.save_results,
            f"mse_backward_rollout_{self.extra_note}_run{self.run}.txt",
        )

        np.savetxt(mse_save_path, mse_backward.numpy())

        print(f"Saved backward rollout MSE to: {mse_save_path}")

        return real_backward, gen_backward, mse_backward

    def _cyclic_rollout_once(self, candidate_idx=None, verbose=True):
        vids_subset = self.vids_test[: self.num_traj_rollout]

        N, T, C, H, W = vids_subset.shape
        T_roll = min(T, self.max_T_rollout)

        if T_roll < 2:
            raise ValueError(f"T_roll must be at least 2 for cyclic rollout, got {T_roll}")

        fwd_traj = torch.zeros(N, T_roll, C, H, W)
        bwd_traj = torch.zeros(N, T_roll, C, H, W)

        fwd_traj[:, 0] = vids_subset[:, 0].detach().cpu()
        cond = fwd_traj[:, 0]

        for t_idx in range(1, T_roll):
            path = self.model.sample(
                x_cond=cond,
                num_ode_steps=self.num_ode_steps,
                ddim_eta=self.ddim_eta,
                c=self.c_forward,
            )

            x_next = path[:, -1].detach().cpu()
            fwd_traj[:, t_idx] = x_next
            cond = x_next

            if verbose:
                prefix = "" if candidate_idx is None else f"Candidate {candidate_idx}: "
                print(f"{prefix}Cyclic forward step {t_idx}/{T_roll - 1}")

        bwd_traj[:, 0] = fwd_traj[:, -1]
        cond = bwd_traj[:, 0]

        for t_idx in range(1, T_roll):
            path = self.model.sample(
                x_cond=cond,
                num_ode_steps=self.num_ode_steps,
                ddim_eta=self.ddim_eta,
                c=self.c_backward,
            )

            x_prev = path[:, -1].detach().cpu()
            bwd_traj[:, t_idx] = x_prev
            cond = x_prev

            if verbose:
                prefix = "" if candidate_idx is None else f"Candidate {candidate_idx}: "
                print(f"{prefix}Cyclic backward step {t_idx}/{T_roll - 1}")

        endpoint_error = ((bwd_traj[:, -1] - fwd_traj[:, 0]) ** 2).mean(dim=(1, 2, 3))

        fwd_reversed = torch.flip(fwd_traj, dims=[1])
        path_error = ((bwd_traj - fwd_reversed) ** 2).mean(dim=(1, 2, 3, 4))

        return fwd_traj, bwd_traj, endpoint_error, path_error

    def cyclic_rollout(self):
        fwd_traj, bwd_traj, endpoint_error, path_error = self._cyclic_rollout_once(
            candidate_idx=None,
            verbose=True,
        )

        save_path = os.path.join(
            self.save_results,
            f"cyclic_results_{self.extra_note}_run{self.run}.pt",
        )

        torch.save(
            {
                "fwd_traj": fwd_traj,
                "bwd_traj": bwd_traj,
                "endpoint_error": endpoint_error,
                "path_error": path_error,
                "c_forward": self.c_forward,
                "c_backward": self.c_backward,
            },
            save_path,
        )

        print(f"Saved cyclic rollout results to: {save_path}")
        print(f"Mean endpoint cyclic error: {endpoint_error.mean().item():.6f}")
        print(f"Mean path-wise cyclic error: {path_error.mean().item():.6f}")

        return fwd_traj, bwd_traj, endpoint_error, path_error

    def cyclic_rollout_multi(self):
        """
        Multi-candidate cyclic inference.

        For each K in self.cyclic_runs, run K stochastic cyclic rollouts:
            X0 -> forward rollout -> XT_hat
            XT_hat -> backward rollout -> X0_recon

        Records:
            mean endpoint/path error across candidates
            best endpoint/path candidate
            total inference time
            time per candidate
        """

        all_group_results = {}

        for K in self.cyclic_runs:
            K = int(K)

            if K <= 0:
                raise ValueError(f"cyclic_runs entries must be positive, got {K}")

            print(f"\n=== Multi-candidate cyclic inference: K={K} ===")

            group_start_time = time.time()

            fwd_trajs = []
            bwd_trajs = []
            endpoint_errors = []
            path_errors = []
            candidate_times = []

            for candidate_idx in range(K):
                candidate_start_time = time.time()

                verbose = K <= 4

                fwd_traj, bwd_traj, endpoint_error, path_error = self._cyclic_rollout_once(
                    candidate_idx=candidate_idx,
                    verbose=verbose,
                )

                candidate_time = time.time() - candidate_start_time

                fwd_trajs.append(fwd_traj)
                bwd_trajs.append(bwd_traj)
                endpoint_errors.append(endpoint_error)
                path_errors.append(path_error)
                candidate_times.append(candidate_time)

                print(
                    f"K={K} candidate {candidate_idx + 1}/{K} | "
                    f"endpoint={endpoint_error.mean().item():.6f} | "
                    f"path={path_error.mean().item():.6f} | "
                    f"time={candidate_time:.2f}s"
                )

            total_time = time.time() - group_start_time

            endpoint_errors_tensor = torch.stack(endpoint_errors, dim=0)
            path_errors_tensor = torch.stack(path_errors, dim=0)
            candidate_times_tensor = torch.tensor(candidate_times, dtype=torch.float32)

            endpoint_mean_per_candidate = endpoint_errors_tensor.mean(dim=1)
            path_mean_per_candidate = path_errors_tensor.mean(dim=1)

            best_endpoint_idx = int(torch.argmin(endpoint_mean_per_candidate).item())
            best_path_idx = int(torch.argmin(path_mean_per_candidate).item())

            mean_endpoint = endpoint_mean_per_candidate.mean()
            mean_path = path_mean_per_candidate.mean()

            best_endpoint = endpoint_mean_per_candidate[best_endpoint_idx]
            best_path = path_mean_per_candidate[best_path_idx]

            summary = {
                "K": K,
                "mean_endpoint_error": mean_endpoint,
                "mean_path_error": mean_path,
                "best_endpoint_error": best_endpoint,
                "best_path_error": best_path,
                "best_endpoint_idx": best_endpoint_idx,
                "best_path_idx": best_path_idx,
                "total_time_sec": total_time,
                "time_per_candidate_sec": total_time / K,
                "candidate_times_sec": candidate_times_tensor,
            }

            save_path = os.path.join(
                self.save_results,
                f"cyclic_multi_K{K}_{self.extra_note}_run{self.run}.pt",
            )

            torch.save(
                {
                    "summary": summary,
                    "endpoint_errors": endpoint_errors_tensor,
                    "path_errors": path_errors_tensor,
                    "candidate_times_sec": candidate_times_tensor,
                    "best_endpoint_fwd_traj": fwd_trajs[best_endpoint_idx],
                    "best_endpoint_bwd_traj": bwd_trajs[best_endpoint_idx],
                    "best_path_fwd_traj": fwd_trajs[best_path_idx],
                    "best_path_bwd_traj": bwd_trajs[best_path_idx],
                    "c_forward": self.c_forward,
                    "c_backward": self.c_backward,
                },
                save_path,
            )

            summary_path = os.path.join(
                self.save_results,
                f"cyclic_multi_summary_K{K}_{self.extra_note}_run{self.run}.txt",
            )

            with open(summary_path, "w") as f:
                f.write(f"K: {K}\n")
                f.write(f"Mean endpoint cyclic error: {mean_endpoint.item():.8f}\n")
                f.write(f"Best endpoint cyclic error: {best_endpoint.item():.8f}\n")
                f.write(f"Best endpoint candidate idx: {best_endpoint_idx}\n")
                f.write(f"Mean path-wise cyclic error: {mean_path.item():.8f}\n")
                f.write(f"Best path-wise cyclic error: {best_path.item():.8f}\n")
                f.write(f"Best path candidate idx: {best_path_idx}\n")
                f.write(f"Total inference time sec: {total_time:.4f}\n")
                f.write(f"Time per candidate sec: {total_time / K:.4f}\n")
                f.write("Candidate times sec:\n")
                for idx, t_val in enumerate(candidate_times):
                    f.write(f"  candidate {idx}: {t_val:.4f}\n")

            print(f"Saved multi-cyclic results to: {save_path}")
            print(f"Saved multi-cyclic summary to: {summary_path}")
            print(f"K={K} mean endpoint cyclic error: {mean_endpoint.item():.6f}")
            print(f"K={K} best endpoint cyclic error: {best_endpoint.item():.6f}")
            print(f"K={K} mean path-wise cyclic error: {mean_path.item():.6f}")
            print(f"K={K} best path-wise cyclic error: {best_path.item():.6f}")
            print(f"K={K} total inference time: {total_time:.2f}s")
            print(f"K={K} time per candidate: {total_time / K:.2f}s")

            all_group_results[K] = {
                "summary": summary,
                "endpoint_errors": endpoint_errors_tensor,
                "path_errors": path_errors_tensor,
                "save_path": save_path,
                "summary_path": summary_path,
            }

        return all_group_results

    def run_all(self, frame_idx):
        real_nsp, pred_nsp, mse_nsp = self.evaluate_next_step()
        real_prev, pred_prev, mse_prev = self.evaluate_previous_step()

        real_single, pred_single, mse_single = self.sample_single_step(frame_idx)

        backward_frame_idx = max(1, frame_idx + 1)
        real_single_b, pred_single_b, mse_single_b = self.sample_single_step_backward(
            backward_frame_idx
        )

        real_rollout, gen_rollout, mse_rollout = self.rollout_and_visualize()

        real_backward, gen_backward, mse_backward = self.backward_rollout_and_visualize()

        cyclic_multi_results = self.cyclic_rollout_multi()

        first_K = int(self.cyclic_runs[0])
        first_result = cyclic_multi_results[first_K]

        endpoint_error = first_result["endpoint_errors"][0]
        path_error = first_result["path_errors"][0]

        return {
            "real_nsp": real_nsp,
            "pred_nsp": pred_nsp,
            "mse_nsp": mse_nsp,
            "real_prev": real_prev,
            "pred_prev": pred_prev,
            "mse_prev": mse_prev,
            "real_single": real_single,
            "pred_single": pred_single,
            "mse_single": mse_single,
            "real_single_backward": real_single_b,
            "pred_single_backward": pred_single_b,
            "mse_single_backward": mse_single_b,
            "real_rollout": real_rollout,
            "gen_rollout": gen_rollout,
            "mse_rollout": mse_rollout,
            "real_backward": real_backward,
            "gen_backward": gen_backward,
            "mse_backward": mse_backward,
            "endpoint_cyclic_error": endpoint_error,
            "path_cyclic_error": path_error,
            "cyclic_multi_results": cyclic_multi_results,
        }
