#!/usr/bin/env python3

"""Run the existing DiT cyclic-evaluation script R independent times.

Each run keeps all outputs from generate_tensors_plot_error_bar_dit.py in a
separate directory. The driver then writes per-run and across-run summaries.
"""

import argparse
import csv
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch

from generate_tensors_plot_error_bar_dit import main as single_run_main


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_result_pt(run_dir):
    matches = sorted(Path(run_dir).glob("dataset_cyclic_vs_runs_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one result .pt in {run_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def summarize_result(data):
    K_values = [int(k) for k in data["K_values"]]
    true_errors = data["true_errors_all_candidates"].float()
    selected = data["selected_true_error_by_K"].float()

    random_by_K = torch.stack(
        [true_errors[:K].mean(dim=0) for K in K_values], dim=0
    )
    oracle_by_K = torch.stack(
        [true_errors[:K].min(dim=0).values for K in K_values], dim=0
    )

    random_mean = random_by_K.mean(dim=1)
    random_std = random_by_K.std(dim=1, unbiased=False)
    cyclic_mean = selected.mean(dim=1)
    cyclic_std = selected.std(dim=1, unbiased=False)
    oracle_mean = oracle_by_K.mean(dim=1)
    oracle_std = oracle_by_K.std(dim=1, unbiased=False)
    delta_cyc = 100.0 * (random_mean - cyclic_mean) / random_mean

    return {
        "K_values": K_values,
        "path_mean": data["path_mean"].float(),
        "path_std": data["path_std"].float(),
        "endpoint_mean": data["endpoint_mean"].float(),
        "endpoint_std": data["endpoint_std"].float(),
        "random_mean": random_mean,
        "random_std": random_std,
        "cyclic_mean": cyclic_mean,
        "cyclic_std": cyclic_std,
        "oracle_mean": oracle_mean,
        "oracle_std": oracle_std,
        "delta_cyc": delta_cyc,
    }


def run_single(args, run_dir, seed):
    argv = [
        "generate_tensors_plot_error_bar_dit.py",
        "--ckpt", args.ckpt,
        "--dataset", args.dataset,
        "--samp_diff_steps", str(args.samp_diff_steps),
        "--ddim_eta", str(args.ddim_eta),
        "--max_T_rollout", str(args.max_T_rollout),
        "--K_values", *[str(k) for k in args.K_values],
        "--device_idx", str(args.device_idx),
        "--traj_batch_size", str(args.traj_batch_size),
        "--out_dir", str(run_dir),
    ]

    old_argv = sys.argv
    try:
        set_seed(seed)
        sys.argv = argv
        single_run_main()
    finally:
        sys.argv = old_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_per_run_csv(path, runs):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run", "seed", "K",
            "path_mean", "path_std",
            "endpoint_mean", "endpoint_std",
            "E_random", "E_random_dataset_std",
            "E_cyclic", "E_cyclic_dataset_std",
            "E_oracle", "E_oracle_dataset_std",
            "delta_cyc_percent",
        ])
        for run in runs:
            s = run["summary"]
            for i, K in enumerate(s["K_values"]):
                writer.writerow([
                    run["run"], run["seed"], K,
                    s["path_mean"][i].item(), s["path_std"][i].item(),
                    s["endpoint_mean"][i].item(), s["endpoint_std"][i].item(),
                    s["random_mean"][i].item(), s["random_std"][i].item(),
                    s["cyclic_mean"][i].item(), s["cyclic_std"][i].item(),
                    s["oracle_mean"][i].item(), s["oracle_std"][i].item(),
                    s["delta_cyc"][i].item(),
                ])


def stack_metric(runs, name):
    return torch.stack([r["summary"][name] for r in runs], dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Run DiT cyclic inference R times and aggregate metrics."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="datasets/vids_test_ce-rm.npy")
    parser.add_argument("--samp_diff_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--max_T_rollout", type=int, default=20)
    parser.add_argument("--K_values", type=int, nargs="+", default=[8])
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--traj_batch_size", type=int, default=1)
    parser.add_argument("--R", type=int, default=10, help="Number of independent inference runs.")
    parser.add_argument("--base_seed", type=int, default=2027)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/dit_cyclic_vs_runs_multirun",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed run directories instead of reusing them.",
    )
    args = parser.parse_args()

    if args.R <= 0:
        raise ValueError("--R must be positive.")
    if args.samp_diff_steps <= 0:
        raise ValueError("--samp_diff_steps must be positive.")
    if args.traj_batch_size <= 0:
        raise ValueError("--traj_batch_size must be positive.")
    if any(K <= 0 for K in args.K_values):
        raise ValueError("Every entry in --K_values must be positive.")

    args.K_values = sorted(set(args.K_values))
    ckpt_stem = Path(args.ckpt).stem
    k_tag = "-".join(str(k) for k in args.K_values)
    eta_tag = f"{args.ddim_eta:g}"
    exp_name = (
        f"{ckpt_stem}_diffsteps{args.samp_diff_steps}_eta{eta_tag}_"
        f"K{k_tag}_R{args.R}"
    )
    exp_dir = Path(args.out_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_idx in range(args.R):
        run_number = run_idx + 1
        seed = args.base_seed + run_idx
        run_dir = exp_dir / f"run_{run_number:02d}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(run_dir.glob("dataset_cyclic_vs_runs_*.pt"))
        if existing and not args.overwrite:
            if len(existing) != 1:
                raise RuntimeError(f"Found multiple .pt results in {run_dir}: {existing}")
            pt_path = existing[0]
            print(f"\n===== RUN {run_number}/{args.R} | seed={seed} | REUSING {pt_path} =====")
        else:
            print(f"\n===== RUN {run_number}/{args.R} | seed={seed} =====")
            run_single(args, run_dir, seed)
            pt_path = find_result_pt(run_dir)

        summary = summarize_result(load_pt(pt_path))
        runs.append({"run": run_number, "seed": seed, "pt_path": str(pt_path), "summary": summary})

    K_values = runs[0]["summary"]["K_values"]
    for r in runs[1:]:
        if r["summary"]["K_values"] != K_values:
            raise RuntimeError("K_values differ across runs.")

    per_run_csv = exp_dir / "per_run_metrics.csv"
    write_per_run_csv(per_run_csv, runs)

    metric_names = [
        "path_mean", "endpoint_mean",
        "random_mean", "cyclic_mean", "oracle_mean", "delta_cyc",
    ]
    stacked = {name: stack_metric(runs, name) for name in metric_names}
    across_mean = {name: x.mean(dim=0) for name, x in stacked.items()}
    across_std = {
        name: x.std(dim=0, unbiased=(args.R > 1)) for name, x in stacked.items()
    }

    delta_from_mean_errors = (
        100.0
        * (across_mean["random_mean"] - across_mean["cyclic_mean"])
        / across_mean["random_mean"]
    )

    aggregate_csv = exp_dir / "average_metrics_across_runs.csv"
    with open(aggregate_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "K", "R",
            "path_mean", "path_run_std",
            "endpoint_mean", "endpoint_run_std",
            "E_random_mean", "E_random_run_std",
            "E_cyclic_mean", "E_cyclic_run_std",
            "E_oracle_mean", "E_oracle_run_std",
            "delta_cyc_mean_of_runs_percent", "delta_cyc_run_std_percent",
            "delta_cyc_from_mean_errors_percent",
        ])
        for i, K in enumerate(K_values):
            writer.writerow([
                K, args.R,
                across_mean["path_mean"][i].item(), across_std["path_mean"][i].item(),
                across_mean["endpoint_mean"][i].item(), across_std["endpoint_mean"][i].item(),
                across_mean["random_mean"][i].item(), across_std["random_mean"][i].item(),
                across_mean["cyclic_mean"][i].item(), across_std["cyclic_mean"][i].item(),
                across_mean["oracle_mean"][i].item(), across_std["oracle_mean"][i].item(),
                across_mean["delta_cyc"][i].item(), across_std["delta_cyc"][i].item(),
                delta_from_mean_errors[i].item(),
            ])

    aggregate_pt = exp_dir / "multirun_metrics.pt"
    torch.save({
        "R": args.R,
        "base_seed": args.base_seed,
        "seeds": [r["seed"] for r in runs],
        "K_values": K_values,
        "run_result_paths": [r["pt_path"] for r in runs],
        "metrics_by_run": stacked,
        "mean_across_runs": across_mean,
        "std_across_runs": across_std,
        "delta_cyc_from_mean_errors": delta_from_mean_errors,
        "checkpoint": args.ckpt,
        "dataset": args.dataset,
        "samp_diff_steps": args.samp_diff_steps,
        "ddim_eta": args.ddim_eta,
        "max_T_rollout": args.max_T_rollout,
    }, aggregate_pt)

    summary_txt = exp_dir / "average_metrics_across_runs.txt"
    lines = [
        "===== DIT MULTI-RUN SUMMARY =====",
        f"R = {args.R}",
        f"Seeds = {[r['seed'] for r in runs]}",
        f"Checkpoint = {args.ckpt}",
        f"Dataset = {args.dataset}",
        f"Sampling diffusion steps = {args.samp_diff_steps}",
        f"DDIM eta = {args.ddim_eta}",
    ]
    for i, K in enumerate(K_values):
        lines += [
            "",
            f"K = {K}",
            f"E_random = {across_mean['random_mean'][i].item():.6f} +/- {across_std['random_mean'][i].item():.6f} across runs",
            f"E_cyclic = {across_mean['cyclic_mean'][i].item():.6f} +/- {across_std['cyclic_mean'][i].item():.6f} across runs",
            f"E_oracle = {across_mean['oracle_mean'][i].item():.6f} +/- {across_std['oracle_mean'][i].item():.6f} across runs",
            f"Delta_cyc mean(run-wise) = {across_mean['delta_cyc'][i].item():.2f}% +/- {across_std['delta_cyc'][i].item():.2f}%",
            f"Delta_cyc from averaged E values = {delta_from_mean_errors[i].item():.2f}%",
            f"Path-wise cyclic loss = {across_mean['path_mean'][i].item():.6f} +/- {across_std['path_mean'][i].item():.6f} across runs",
            f"Endpoint cyclic loss = {across_mean['endpoint_mean'][i].item():.6f} +/- {across_std['endpoint_mean'][i].item():.6f} across runs",
        ]
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nSaved per-run metrics: {per_run_csv}")
    print(f"Saved average metrics: {aggregate_csv}")
    print(f"Saved aggregate tensors: {aggregate_pt}")
    print(f"Saved text summary: {summary_txt}")


if __name__ == "__main__":
    main()
