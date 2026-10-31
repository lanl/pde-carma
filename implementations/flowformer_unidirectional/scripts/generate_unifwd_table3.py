#!/usr/bin/env python3
"""Evaluate a forward-only FlowFormer under the Table-3 Forward protocol.

For each independent realization r, generate K stochastic autoregressive
candidates for every test trajectory.  E_fwd^(r) is the mean rollout MSE over
all K candidates and all test trajectories.  The reported Uni-Fwd value is the
mean +/- sample standard deviation of E_fwd^(r) over R realizations.

The observed initial frame t=0 is excluded from rollout MSE, matching the
current bidirectional candidate-pool evaluation script.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.device_manager import DeviceManager
from utils.my_model import FlowFormerInference


def resolve_path(path):
    if os.path.isfile(path):
        return os.path.abspath(path)
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find file: {path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_forward(model, x0, T_roll, num_ode_steps):
    """Generate one stochastic autoregressive rollout from x0."""
    N, C, H, W = x0.shape
    trajectory = torch.empty(N, T_roll, C, H, W, dtype=torch.float32)
    trajectory[:, 0] = x0.detach().cpu()
    condition = trajectory[:, 0]

    for t in range(1, T_roll):
        path = model.sample(
            x_cond=condition,
            num_ode_steps=num_ode_steps,
        )
        x_next = path[:, -1].detach().cpu()
        trajectory[:, t] = x_next
        condition = x_next

    return trajectory


def rollout_mse(predicted, ground_truth):
    """Per-trajectory full-horizon MSE, excluding the observed t=0 frame."""
    return ((predicted[:, 1:] - ground_truth[:, 1:]) ** 2).mean(
        dim=(1, 2, 3, 4)
    )


def evaluate_one_realization(
    model,
    videos_test,
    K,
    T_roll,
    num_ode_steps,
    traj_batch_size,
):
    N_total = videos_test.shape[0]
    true_errors = torch.empty(K, N_total, dtype=torch.float32)

    for candidate_idx in range(K):
        print(f"  Candidate {candidate_idx + 1}/{K}")
        for start in range(0, N_total, traj_batch_size):
            end = min(start + traj_batch_size, N_total)
            real = videos_test[start:end, :T_roll].detach().cpu()
            pred = rollout_forward(
                model=model,
                x0=real[:, 0],
                T_roll=T_roll,
                num_ode_steps=num_ode_steps,
            )
            true_errors[candidate_idx, start:end] = rollout_mse(pred, real)

    # Eq. (10) in the paper: average candidates per trajectory, then test set.
    per_trajectory_forward = true_errors.mean(dim=0)
    E_fwd = per_trajectory_forward.mean()
    return E_fwd, true_errors


def main():
    parser = argparse.ArgumentParser(
        description="Compute the Uni-Fwd entry for Table 3 from a forward-only checkpoint."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=2027)
    parser.add_argument("--num_ode_steps", type=int, default=5)
    parser.add_argument("--max_T_rollout", type=int, default=20)
    parser.add_argument("--traj_batch_size", type=int, default=4)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/unifwd_table3")
    args = parser.parse_args()

    if args.K <= 0 or args.R <= 0:
        raise ValueError("--K and --R must be positive.")
    if args.num_ode_steps <= 0 or args.max_T_rollout < 2:
        raise ValueError("--num_ode_steps must be positive and --max_T_rollout >= 2.")
    if args.traj_batch_size <= 0:
        raise ValueError("--traj_batch_size must be positive.")

    ckpt_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint["args"]
    if not isinstance(config, dict):
        config = vars(config)

    direction = config.get("direction", "forward")
    if direction != "forward":
        raise ValueError(
            f"Table-3 Uni-Fwd requires a forward checkpoint, got direction={direction!r}."
        )

    videos_test = torch.from_numpy(np.load(dataset_path, mmap_mode="r")).float()
    if videos_test.ndim != 5:
        raise ValueError(
            f"Expected test data with shape (N,T,C,H,W), got {tuple(videos_test.shape)}."
        )

    N_total, T_total, C, H, W = videos_test.shape
    T_roll = min(args.max_T_rollout, T_total)

    devices = DeviceManager.list_devices()
    if args.device_idx >= len(devices):
        raise IndexError(
            f"device_idx={args.device_idx} but available devices are {devices}."
        )
    device = devices[args.device_idx]

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=ckpt_path,
        img_size=(H, W),
        device=device,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_stem = Path(ckpt_path).stem
    exp_dir = out_dir / f"{ckpt_stem}_K{args.K}_R{args.R}_ode{args.num_ode_steps}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_values = []
    all_errors = []

    print("===================================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Test shape: {tuple(videos_test.shape)}")
    print(f"K={args.K}, R={args.R}, T_roll={T_roll}, ODE steps={args.num_ode_steps}")
    print(f"Device: {device}")
    print("===================================")

    for run_idx in range(args.R):
        seed = args.base_seed + run_idx
        set_seed(seed)
        print(f"\n===== REALIZATION {run_idx + 1}/{args.R} | seed={seed} =====")
        E_fwd, true_errors = evaluate_one_realization(
            model=model,
            videos_test=videos_test,
            K=args.K,
            T_roll=T_roll,
            num_ode_steps=args.num_ode_steps,
            traj_batch_size=args.traj_batch_size,
        )
        run_values.append(E_fwd.cpu())
        all_errors.append(true_errors)
        print(f"E_fwd^(r) = {E_fwd.item():.10f}")

    run_values = torch.stack(run_values)
    mean = run_values.mean()
    std = run_values.std(unbiased=(args.R > 1))
    all_errors = torch.stack(all_errors, dim=0)  # (R,K,N)

    per_run_csv = exp_dir / "per_run_metrics.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "K", "E_fwd"])
        for i, value in enumerate(run_values):
            writer.writerow([i + 1, args.base_seed + i, args.K, value.item()])

    summary_csv = exp_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "dataset", "K", "R", "num_ode_steps", "T_roll",
            "num_test_trajectories", "E_fwd_mean", "E_fwd_run_std",
        ])
        writer.writerow([
            ckpt_path, dataset_path, args.K, args.R, args.num_ode_steps, T_roll,
            N_total, mean.item(), std.item(),
        ])

    pt_path = exp_dir / "unifwd_table3_metrics.pt"
    torch.save(
        {
            "checkpoint": ckpt_path,
            "dataset": dataset_path,
            "K": args.K,
            "R": args.R,
            "base_seed": args.base_seed,
            "seeds": [args.base_seed + i for i in range(args.R)],
            "num_ode_steps": args.num_ode_steps,
            "T_roll": T_roll,
            "num_test_trajectories": N_total,
            "E_fwd_by_run": run_values,
            "E_fwd_mean": mean,
            "E_fwd_run_std": std,
            "true_errors_by_run": all_errors,
        },
        pt_path,
    )

    summary_txt = exp_dir / "summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "===== TABLE 3 UNI-FWD =====",
                f"checkpoint = {ckpt_path}",
                f"dataset = {dataset_path}",
                f"K = {args.K}",
                f"R = {args.R}",
                f"seeds = {[args.base_seed + i for i in range(args.R)]}",
                f"ODE steps = {args.num_ode_steps}",
                f"T_roll = {T_roll}",
                f"N_test = {N_total}",
                f"Uni-Fwd = {mean.item():.10f} +/- {std.item():.10f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== TABLE 3 RESULT =====")
    print(f"Uni-Fwd = {mean.item():.10f} +/- {std.item():.10f}")
    print(f"Saved: {per_run_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {pt_path}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
