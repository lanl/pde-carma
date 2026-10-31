#!/usr/bin/env python3
"""Table-3 Uni-Fwd evaluation for a forward-only DiT checkpoint."""

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
from utils.my_model import DiTInference


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return os.path.abspath(path)
    path = os.path.join(PROJECT_ROOT, "models", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return os.path.abspath(path)


def rollout_forward(model, x0, T_roll, samp_diff_steps, ddim_eta):
    """Generate one autoregressive forward candidate."""
    n, c, h, w = x0.shape
    pred = torch.empty(n, T_roll, c, h, w, dtype=torch.float32)
    pred[:, 0] = x0.detach().cpu()
    cond = pred[:, 0]
    c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)

    for t in range(1, T_roll):
        cond = model.sample(
            x_cond=cond,
            samp_diff_steps=samp_diff_steps,
            c=c_forward,
            ddim_eta=ddim_eta,
            return_path=False,
        ).detach().cpu()
        pred[:, t] = cond

    return pred


def rollout_mse(pred, real):
    """Paper metric: MSE over future frames t=1,...,T."""
    return ((pred[:, 1:] - real[:, 1:]) ** 2).mean(dim=(1, 2, 3, 4))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Uni-Fwd DiT with K candidates and R realizations."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=2027)
    parser.add_argument("--samp_diff_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--max_T_rollout", type=int, default=20)
    parser.add_argument("--traj_batch_size", type=int, default=1)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=str, default="results/table3_unidirectional"
    )
    args = parser.parse_args()

    if args.K <= 0 or args.R <= 0 or args.traj_batch_size <= 0:
        raise ValueError("K, R, and traj_batch_size must be positive.")

    ckpt_path = resolve_checkpoint(args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint["args"]
    if not isinstance(config, dict):
        config = vars(config)

    ds = config["ds"]
    ms = config["ms"]
    dit_arch = config.get("dit_arch", "legacy")
    direction = config.get("direction")
    if direction not in {None, "forward"}:
        raise ValueError(f"Expected forward checkpoint, got direction={direction!r}")

    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_root = args.dataset_root or PROJECT_ROOT
        dataset_path = os.path.join(
            dataset_root, "datasets", f"vids_test_{ds}.npy"
        )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Test data not found: {dataset_path}")

    vids_test = torch.from_numpy(np.load(dataset_path)).float()
    if vids_test.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W], got {tuple(vids_test.shape)}")

    N, T, _, H, W = vids_test.shape
    T_roll = min(T, args.max_T_rollout)
    if T_roll < 2:
        raise ValueError("Rollout horizon must contain at least one future frame.")

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]
    model = DiTInference.from_pretrained(ckpt_path, (H, W), device=device)

    print("===================================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {dit_arch}-{ms}, direction=forward")
    print(f"Test shape: {tuple(vids_test.shape)}")
    print(f"K={args.K}, R={args.R}, T_roll={T_roll}")
    print(f"Sampling steps={args.samp_diff_steps}, DDIM eta={args.ddim_eta}")
    print("===================================")

    e_fwd_runs = []
    true_errors_runs = []

    with torch.inference_mode():
        for r in range(args.R):
            seed = args.base_seed + r
            set_seed(seed)
            true_errors = torch.empty(args.K, N, dtype=torch.float32)
            print(f"\n===== Realization {r + 1}/{args.R} | seed={seed} =====")

            for k in range(args.K):
                print(f"Candidate {k + 1}/{args.K}")
                for start in range(0, N, args.traj_batch_size):
                    end = min(start + args.traj_batch_size, N)
                    real = vids_test[start:end, :T_roll].detach().cpu()
                    pred = rollout_forward(
                        model,
                        real[:, 0],
                        T_roll,
                        args.samp_diff_steps,
                        args.ddim_eta,
                    )
                    true_errors[k, start:end] = rollout_mse(pred, real)

            # E_fwd^(r) = mean over test trajectories and K candidates.
            e_fwd = true_errors.mean()
            e_fwd_runs.append(e_fwd)
            true_errors_runs.append(true_errors)
            print(f"Realization E_fwd: {e_fwd.item():.10f}")

    e_fwd_runs = torch.stack(e_fwd_runs)
    e_fwd_mean = e_fwd_runs.mean()
    e_fwd_std = e_fwd_runs.std(unbiased=(args.R > 1))
    true_errors_runs = torch.stack(true_errors_runs)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"dit_{dit_arch}_{ms}_{ds}_unidirectional_K{args.K}_R{args.R}_steps{args.samp_diff_steps}"
    pt_path = out_dir / f"{base}.pt"
    csv_path = out_dir / f"{base}.csv"
    txt_path = out_dir / f"{base}.txt"

    torch.save(
        {
            "checkpoint": ckpt_path,
            "dataset": os.path.abspath(dataset_path),
            "K": args.K,
            "R": args.R,
            "T_roll": T_roll,
            "E_fwd_by_run": e_fwd_runs,
            "E_fwd_mean": e_fwd_mean,
            "E_fwd_run_std": e_fwd_std,
            "true_errors_by_run_candidate_trajectory": true_errors_runs,
        },
        pt_path,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "E_fwd"])
        for r, value in enumerate(e_fwd_runs):
            writer.writerow([r + 1, args.base_seed + r, value.item()])

    with open(txt_path, "w") as f:
        f.write(f"E_fwd_mean: {e_fwd_mean.item():.10f}\n")
        f.write(f"E_fwd_run_std: {e_fwd_std.item():.10f}\n")
        f.write(
            f"table_entry: {e_fwd_mean.item():.10f} +/- {e_fwd_std.item():.10f}\n"
        )

    print("\n===== FINAL UNI-FWD RESULT =====")
    print(f"E_fwd = {e_fwd_mean.item():.10f} +/- {e_fwd_std.item():.10f}")
    print(f"Saved: {pt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
