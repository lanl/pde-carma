#!/usr/bin/env python3
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


@dataclass
class BeamState:
    current: torch.Tensor
    rollout: torch.Tensor
    cumulative_score: float
    selected_indices: List[int]
    selected_horizons: List[int]
    selected_cycle_scores: List[float]
    selected_variances: List[float]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_ode_steps", type=int, default=5)
    p.add_argument("--num_transitions", type=int, default=20)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--variance_samples", type=int, default=4)
    p.add_argument("--lookahead_min", type=int, default=2)
    p.add_argument("--lookahead_max", type=int, default=6)
    p.add_argument("--backward_repeats", type=int, default=2)
    p.add_argument("--beam_min", type=int, default=1)
    p.add_argument("--beam_mid", type=int, default=2)
    p.add_argument("--beam_max", type=int, default=4)
    p.add_argument("--variance_weight", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device_idx", type=int, default=0)
    p.add_argument("--out_dir", default="results/adaptive_beam_cyclic")
    p.add_argument("--traj_indices", type=int, nargs="*", default=None)
    p.add_argument("--all_trajs", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_path(text):
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.inference_mode()
def sample_one_step(model, x_cond, num_ode_steps, direction):
    path = model.sample(
        x_cond=x_cond,
        num_ode_steps=num_ode_steps,
        c=direction,
    )
    return path[:, -1].detach().cpu()


@torch.inference_mode()
def rollout_direction(model, x0, transitions, num_ode_steps, direction):
    b, c, h, w = x0.shape
    traj = torch.empty(b, transitions + 1, c, h, w, dtype=x0.dtype)
    traj[:, 0] = x0.detach().cpu()
    current = traj[:, 0]
    for step in range(transitions):
        current = sample_one_step(
            model, current, num_ode_steps, direction
        )
        traj[:, step + 1] = current
    return traj


@torch.inference_mode()
def estimate_variance(model, candidate, samples, num_ode_steps, c_forward):
    futures = [
        sample_one_step(model, candidate, num_ode_steps, c_forward)
        for _ in range(samples)
    ]
    stack = torch.stack(futures, dim=0)
    return float(stack.var(dim=0, unbiased=False).mean().item())


def normalized_path_cycle_score(forward_path, backward_path, eps):
    reversed_forward = torch.flip(forward_path, dims=[1])
    residual = ((backward_path - reversed_forward) ** 2).mean(dim=(2, 3, 4))
    raw = float(residual[:, 1:].mean().item())
    deltas = forward_path[:, 1:] - forward_path[:, :-1]
    energy = float((deltas ** 2).mean().item())
    return raw / max(energy, eps), raw, energy


def minmax(values, eps):
    lo = min(values)
    hi = max(values)
    scale = max(hi - lo, eps)
    return [(v - lo) / scale for v in values]


def choose_level(current, history, low, mid, high, warmup):
    if len(history) < warmup:
        return mid
    hist = torch.tensor(history, dtype=torch.float64)
    q1 = float(torch.quantile(hist, 0.33).item())
    q2 = float(torch.quantile(hist, 0.67).item())
    if current <= q1:
        return low
    if current >= q2:
        return high
    return mid


@torch.inference_mode()
def score_child(
    model,
    parent_current,
    child,
    horizon,
    backward_repeats,
    num_ode_steps,
    c_forward,
    c_backward,
    eps,
):
    if horizon == 1:
        forward_path = torch.stack([parent_current, child], dim=1)
    else:
        remainder = rollout_direction(
            model,
            child,
            horizon - 1,
            num_ode_steps,
            c_forward,
        )
        forward_path = torch.cat(
            [parent_current.unsqueeze(1), remainder],
            dim=1,
        )

    scores = []
    raws = []
    energies = []

    for _ in range(backward_repeats):
        backward_path = rollout_direction(
            model,
            forward_path[:, -1],
            horizon,
            num_ode_steps,
            c_backward,
        )
        score, raw, energy = normalized_path_cycle_score(
            forward_path,
            backward_path,
            eps,
        )
        scores.append(score)
        raws.append(raw)
        energies.append(energy)

    return float(np.mean(scores)), float(np.mean(raws)), float(np.mean(energies))


@torch.inference_mode()
def run_one(model, real, args, c_forward, c_backward):
    initial = real[:, 0].detach().cpu()
    beams = [
        BeamState(
            current=initial,
            rollout=initial.unsqueeze(1),
            cumulative_score=0.0,
            selected_indices=[],
            selected_horizons=[],
            selected_cycle_scores=[],
            selected_variances=[],
        )
    ]

    uncertainty_history = []
    beam_width_history = []
    horizon_history = []
    oracle_hit_history = []
    local_gain_history = []

    for step in range(args.num_transitions):
        remaining = args.num_transitions - step
        print(
            f"  adaptive beam {step + 1}/{args.num_transitions} "
            f"(parents={len(beams)})",
            flush=True,
        )

        expanded = []
        for parent_idx, parent in enumerate(beams):
            for candidate_idx in range(args.K):
                candidate = sample_one_step(
                    model,
                    parent.current,
                    args.num_ode_steps,
                    c_forward,
                )
                variance = estimate_variance(
                    model,
                    candidate,
                    args.variance_samples,
                    args.num_ode_steps,
                    c_forward,
                )
                expanded.append(
                    {
                        "parent_idx": parent_idx,
                        "candidate_idx": candidate_idx,
                        "candidate": candidate,
                        "variance": variance,
                    }
                )

        step_uncertainty = float(
            np.mean([item["variance"] for item in expanded])
        )

        mid_horizon = round(
            (args.lookahead_min + args.lookahead_max) / 2
        )
        horizon = choose_level(
            step_uncertainty,
            uncertainty_history,
            args.lookahead_min,
            mid_horizon,
            args.lookahead_max,
            args.warmup_steps,
        )
        horizon = min(horizon, remaining)

        beam_width = choose_level(
            step_uncertainty,
            uncertainty_history,
            args.beam_min,
            args.beam_mid,
            args.beam_max,
            args.warmup_steps,
        )

        uncertainty_history.append(step_uncertainty)
        beam_width_history.append(beam_width)
        horizon_history.append(horizon)

        for item in expanded:
            parent = beams[item["parent_idx"]]
            score, raw, energy = score_child(
                model,
                parent.current,
                item["candidate"],
                horizon,
                args.backward_repeats,
                args.num_ode_steps,
                c_forward,
                c_backward,
                args.eps,
            )
            item["cycle_score"] = score
            item["raw_cycle"] = raw
            item["energy"] = energy

        cycle_rank = minmax(
            [item["cycle_score"] for item in expanded],
            args.eps,
        )
        variance_rank = minmax(
            [item["variance"] for item in expanded],
            args.eps,
        )

        true_next = real[:, step + 1]
        true_errors = [
            float(
                ((item["candidate"] - true_next) ** 2).mean().item()
            )
            for item in expanded
        ]
        oracle_idx = int(np.argmin(true_errors))
        avg_true = float(np.mean(true_errors))

        children = []
        for idx, item in enumerate(expanded):
            parent = beams[item["parent_idx"]]
            increment = (
                (1.0 - args.variance_weight) * cycle_rank[idx]
                + args.variance_weight * variance_rank[idx]
            )
            rollout = torch.cat(
                [parent.rollout, item["candidate"].unsqueeze(1)],
                dim=1,
            )
            children.append(
                BeamState(
                    current=item["candidate"],
                    rollout=rollout,
                    cumulative_score=parent.cumulative_score + increment,
                    selected_indices=parent.selected_indices
                    + [item["candidate_idx"]],
                    selected_horizons=parent.selected_horizons
                    + [horizon],
                    selected_cycle_scores=parent.selected_cycle_scores
                    + [item["cycle_score"]],
                    selected_variances=parent.selected_variances
                    + [item["variance"]],
                )
            )

        ranking = sorted(
            range(len(children)),
            key=lambda i: children[i].cumulative_score,
        )
        keep = ranking[: min(beam_width, len(children))]
        beams = [children[i] for i in keep]

        best_idx = keep[0]
        oracle_hit = float(best_idx == oracle_idx)
        selected_true = true_errors[best_idx]
        local_gain = (
            avg_true - selected_true
        ) / max(avg_true, args.eps)

        oracle_hit_history.append(oracle_hit)
        local_gain_history.append(local_gain)

        print(
            f"    uncertainty={step_uncertainty:.6g}, "
            f"horizon={horizon}, beam={beam_width}, "
            f"oracle hit={oracle_hit:.3f}, "
            f"gain={100.0 * local_gain:.2f}%",
            flush=True,
        )

    best = min(beams, key=lambda beam: beam.cumulative_score)

    per_time = ((best.rollout - real) ** 2).mean(dim=(2, 3, 4))
    midpoint = (best.rollout.shape[1] - 1) // 2

    return {
        "tau0": float(per_time[:, 0].item()),
        "tau_mid": float(per_time[:, midpoint].item()),
        "tau_final": float(per_time[:, -1].item()),
        "rollout_mean": float(per_time.mean().item()),
        "oracle_hit_rate": float(np.mean(oracle_hit_history)),
        "mean_local_gain": float(np.mean(local_gain_history)),
        "mean_beam_width": float(np.mean(beam_width_history)),
        "mean_horizon": float(np.mean(horizon_history)),
        "final_cumulative_score": float(best.cumulative_score),
        "selected_indices": best.selected_indices,
        "selected_horizons": best.selected_horizons,
        "selected_cycle_scores": best.selected_cycle_scores,
        "selected_variances": best.selected_variances,
        "beam_width_history": beam_width_history,
        "horizon_history": horizon_history,
        "uncertainty_history": uncertainty_history,
    }


def aggregate(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def main():
    args = parse_args()

    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.variance_samples < 2:
        raise ValueError("--variance_samples must be >= 2")
    if args.lookahead_min < 1:
        raise ValueError("--lookahead_min must be >= 1")
    if args.lookahead_max < args.lookahead_min:
        raise ValueError("--lookahead_max must be >= --lookahead_min")
    if not 1 <= args.beam_min <= args.beam_mid <= args.beam_max:
        raise ValueError("Require beam_min <= beam_mid <= beam_max")
    if not 0.0 <= args.variance_weight <= 1.0:
        raise ValueError("--variance_weight must be in [0,1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["args"]
    model_size = config["ms"]
    dataset_name = config["ds"]

    videos = np.load(dataset_path, mmap_mode="r")
    n_total, t_total, _, height, width = videos.shape

    required_states = args.num_transitions + 1
    if t_total < required_states:
        raise ValueError(
            f"Dataset has {t_total} states; need {required_states}."
        )

    if args.all_trajs:
        traj_indices = list(range(n_total))
    elif args.traj_indices is None:
        traj_indices = [0, n_total // 2, n_total - 1]
    else:
        traj_indices = args.traj_indices

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)
    c_backward = torch.tensor([0.0, 1.0], dtype=torch.float32)

    print("=" * 88)
    print("Adaptive beam-search cyclic inference")
    print("=" * 88)
    print("Checkpoint:", ckpt_path)
    print("Dataset:", dataset_path)
    print("Device:", device)
    print("K:", args.K)
    print(
        "Beam widths:",
        args.beam_min,
        args.beam_mid,
        args.beam_max,
    )
    print(
        "Lookahead range:",
        args.lookahead_min,
        args.lookahead_max,
    )
    print("=" * 88)

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=str(ckpt_path),
        img_size=(height, width),
        device=device,
    )

    rows = []
    details = []

    for traj_idx in traj_indices:
        print(f"\nTrajectory {traj_idx}", flush=True)

        real = torch.from_numpy(
            np.asarray(
                videos[
                    traj_idx : traj_idx + 1,
                    :required_states,
                ]
            )
        ).float()

        result = run_one(
            model,
            real,
            args,
            c_forward,
            c_backward,
        )

        row = {
            "model": f"flowformer_{model_size}_{dataset_name}",
            "dataset": dataset_name,
            "method": "adaptive_beam_cyclic",
            "trajectory_index": traj_idx,
            "tau0": result["tau0"],
            "tau_mid": result["tau_mid"],
            "tau_final": result["tau_final"],
            "rollout_mean": result["rollout_mean"],
            "oracle_hit_rate": result["oracle_hit_rate"],
            "mean_local_gain": result["mean_local_gain"],
            "mean_beam_width": result["mean_beam_width"],
            "mean_horizon": result["mean_horizon"],
            "final_cumulative_score": result["final_cumulative_score"],
            "K": args.K,
            "beam_min": args.beam_min,
            "beam_mid": args.beam_mid,
            "beam_max": args.beam_max,
            "lookahead_min": args.lookahead_min,
            "lookahead_max": args.lookahead_max,
            "variance_samples": args.variance_samples,
            "backward_repeats": args.backward_repeats,
            "variance_weight": args.variance_weight,
        }
        rows.append(row)

        details.append(
            {
                "trajectory_index": traj_idx,
                "selected_indices": result["selected_indices"],
                "selected_horizons": result["selected_horizons"],
                "selected_cycle_scores": result["selected_cycle_scores"],
                "selected_variances": result["selected_variances"],
                "beam_width_history": result["beam_width_history"],
                "horizon_history": result["horizon_history"],
                "uncertainty_history": result["uncertainty_history"],
            }
        )

    metric_names = [
        "tau0",
        "tau_mid",
        "tau_final",
        "rollout_mean",
        "oracle_hit_rate",
        "mean_local_gain",
        "mean_beam_width",
        "mean_horizon",
    ]

    summary = {
        metric: aggregate([row[metric] for row in rows])
        for metric in metric_names
    }

    print("\n" + "=" * 88)
    print("DATASET-LEVEL SUMMARY")
    print("=" * 88)

    for metric in metric_names:
        item = summary[metric]
        if metric == "mean_local_gain":
            print(
                f"{metric:24s}: "
                f"{100.0 * item['mean']:.8g}% +/- "
                f"{100.0 * item['std']:.8g}%"
            )
        else:
            print(
                f"{metric:24s}: "
                f"{item['mean']:.8g} +/- "
                f"{item['std']:.8g}"
            )

    tag = f"flowformer_{model_size}_{dataset_name}"
    suffix = "all_trajs" if args.all_trajs else "selected_trajs"
    base = (
        f"adaptive_beam_{tag}_K{args.K}_"
        f"B{args.beam_min}-{args.beam_mid}-{args.beam_max}_"
        f"L{args.lookahead_min}-{args.lookahead_max}_"
        f"M{args.variance_samples}_R{args.backward_repeats}_"
        f"V{args.variance_weight:.2f}_"
        f"T{args.num_transitions}_{suffix}"
    )

    csv_path = out_dir / f"{base}.csv"
    pt_path = out_dir / f"{base}.pt"
    txt_path = out_dir / f"{base}.txt"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_args": config,
            "dataset": str(dataset_path),
            "trajectory_indices": traj_indices,
            "args": vars(args),
            "rows": rows,
            "summary": summary,
            "details": details,
        },
        pt_path,
    )

    with txt_path.open("w") as handle:
        handle.write(f"Model: {tag}\n")
        for key, value in vars(args).items():
            handle.write(f"{key}: {value}\n")
        handle.write("\nDataset-level summary\n")
        for metric in metric_names:
            item = summary[metric]
            handle.write(
                f"{metric}: {item['mean']:.8g} +/- "
                f"{item['std']:.8g}\n"
            )
        handle.write("\nLaTeX row\n")
        handle.write(
            f"& Adaptive beam cyclic, $K={args.K}$\n"
            f"& {summary['tau0']['mean']:.6f} $\\pm$ "
            f"{summary['tau0']['std']:.6f}\n"
            f"& {summary['tau_mid']['mean']:.6f} $\\pm$ "
            f"{summary['tau_mid']['std']:.6f}\n"
            f"& {summary['tau_final']['mean']:.6f} $\\pm$ "
            f"{summary['tau_final']['std']:.6f}\n"
            f"& {summary['rollout_mean']['mean']:.6f} $\\pm$ "
            f"{summary['rollout_mean']['std']:.6f} \\\\n"
        )

    print("\nSaved:")
    print(csv_path)
    print(pt_path)
    print(txt_path)


if __name__ == "__main__":
    main()
