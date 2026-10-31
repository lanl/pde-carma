#!/usr/bin/env python3
"""
Short-horizon greedy cyclic selector for bidirectional FlowFormer inference.

At each rollout transition:
  1. Generate K candidate next states.
  2. From each candidate, roll forward for a short lookahead horizon L.
  3. Roll backward from the end of that short path for L steps.
  4. Compare the backward path with the reversed forward path.
  5. Normalize the path-cycle error by the forward-path step energy.
  6. Select the candidate with the smallest normalized score.
  7. Continue from that selected next state.

This script also records:
  - raw one-step greedy cyclic selection,
  - short-horizon normalized greedy cyclic selection,
  - stepwise ground-truth oracle,
so the new selector can be compared fairly.

Example:
  K=4, num_transitions=20, lookahead=4, backward_repeats=2
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short-horizon normalized greedy cyclic FlowFormer inference."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_ode_steps", type=int, default=5)
    parser.add_argument("--num_transitions", type=int, default=20)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument(
        "--lookahead",
        type=int,
        default=4,
        help="Number of forward transitions used to score each candidate.",
    )
    parser.add_argument(
        "--backward_repeats",
        type=int,
        default=2,
        help="Average this many stochastic backward path scores per candidate.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", default="results/strong_greedy_cyclic")
    parser.add_argument("--traj_indices", type=int, nargs="*", default=None)
    parser.add_argument("--all_trajs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.inference_mode()
def sample_one_step(
    model: FlowFormerInference,
    x_cond: torch.Tensor,
    num_ode_steps: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    path = model.sample(
        x_cond=x_cond,
        num_ode_steps=num_ode_steps,
        c=direction,
    )
    return path[:, -1].detach().cpu()


def gather_candidates(
    candidates: torch.Tensor,
    best_idx: torch.Tensor,
) -> torch.Tensor:
    """
    candidates: [K, B, ...]
    best_idx:   [B]
    """
    batch_idx = torch.arange(candidates.shape[1])
    return candidates[best_idx, batch_idx]


@torch.inference_mode()
def rollout_direction(
    model: FlowFormerInference,
    x0: torch.Tensor,
    transitions: int,
    num_ode_steps: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Return [B, transitions+1, C, H, W].
    """
    b, c, h, w = x0.shape
    traj = torch.empty(
        b,
        transitions + 1,
        c,
        h,
        w,
        dtype=x0.dtype,
    )
    traj[:, 0] = x0.detach().cpu()

    current = traj[:, 0]
    for step in range(transitions):
        current = sample_one_step(
            model=model,
            x_cond=current,
            num_ode_steps=num_ode_steps,
            direction=direction,
        )
        traj[:, step + 1] = current

    return traj


def one_step_cycle_score(
    reconstructed_current: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    return ((reconstructed_current - current) ** 2).mean(dim=(1, 2, 3))


def normalized_path_cycle_score(
    forward_path: torch.Tensor,
    backward_path: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    forward_path:  [B, L+1, C, H, W], from current -> future
    backward_path: [B, L+1, C, H, W], from future -> reconstructed current

    Returns:
      normalized_score [B]
      raw_path_score   [B]
      step_energy      [B]
    """
    reversed_forward = torch.flip(forward_path, dims=[1])

    residual = ((backward_path - reversed_forward) ** 2).mean(
        dim=(2, 3, 4)
    )  # [B, L+1]

    # Exclude the exactly shared endpoint at index 0.
    raw_path_score = residual[:, 1:].mean(dim=1)

    forward_deltas = forward_path[:, 1:] - forward_path[:, :-1]
    step_energy = (forward_deltas ** 2).mean(dim=(2, 3, 4)).mean(dim=1)

    normalized_score = raw_path_score / step_energy.clamp_min(eps)

    return normalized_score, raw_path_score, step_energy


@torch.inference_mode()
def run_stepwise_oracle(
    model: FlowFormerInference,
    real: torch.Tensor,
    k: int,
    num_transitions: int,
    num_ode_steps: int,
    c_forward: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    b, _, c, h, w = real.shape
    rollout = torch.empty(
        b, num_transitions + 1, c, h, w, dtype=real.dtype
    )
    rollout[:, 0] = real[:, 0]

    selected_indices = torch.empty(b, num_transitions, dtype=torch.long)
    current = rollout[:, 0]

    for step in range(num_transitions):
        print(
            f"  stepwise oracle {step + 1}/{num_transitions}",
            flush=True,
        )

        candidates = []
        for _ in range(k):
            candidates.append(
                sample_one_step(
                    model=model,
                    x_cond=current,
                    num_ode_steps=num_ode_steps,
                    direction=c_forward,
                )
            )

        candidates_tensor = torch.stack(candidates, dim=0)
        true_next = real[:, step + 1]
        true_errors = ((candidates_tensor - true_next.unsqueeze(0)) ** 2).mean(
            dim=(2, 3, 4)
        )

        best_idx = torch.argmin(true_errors, dim=0)
        selected = gather_candidates(candidates_tensor, best_idx)

        rollout[:, step + 1] = selected
        selected_indices[:, step] = best_idx
        current = selected

    return {
        "rollout": rollout,
        "selected_indices": selected_indices,
    }


@torch.inference_mode()
def run_raw_greedy_cyclic(
    model: FlowFormerInference,
    real: torch.Tensor,
    k: int,
    num_transitions: int,
    num_ode_steps: int,
    c_forward: torch.Tensor,
    c_backward: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    b, _, c, h, w = real.shape
    rollout = torch.empty(
        b, num_transitions + 1, c, h, w, dtype=real.dtype
    )
    rollout[:, 0] = real[:, 0]

    selected_indices = torch.empty(b, num_transitions, dtype=torch.long)
    selected_scores = torch.empty(b, num_transitions)
    oracle_hit = torch.empty(b, num_transitions, dtype=torch.bool)
    local_gain = torch.empty(b, num_transitions)

    current = rollout[:, 0]

    for step in range(num_transitions):
        print(
            f"  raw greedy cyclic {step + 1}/{num_transitions}",
            flush=True,
        )

        forward_candidates = []
        cycle_scores = []

        for _ in range(k):
            candidate = sample_one_step(
                model=model,
                x_cond=current,
                num_ode_steps=num_ode_steps,
                direction=c_forward,
            )
            reconstructed = sample_one_step(
                model=model,
                x_cond=candidate,
                num_ode_steps=num_ode_steps,
                direction=c_backward,
            )
            forward_candidates.append(candidate)
            cycle_scores.append(one_step_cycle_score(reconstructed, current))

        forward_tensor = torch.stack(forward_candidates, dim=0)
        score_tensor = torch.stack(cycle_scores, dim=0)

        true_next = real[:, step + 1]
        true_errors = ((forward_tensor - true_next.unsqueeze(0)) ** 2).mean(
            dim=(2, 3, 4)
        )

        best_idx = torch.argmin(score_tensor, dim=0)
        oracle_idx = torch.argmin(true_errors, dim=0)
        selected = gather_candidates(forward_tensor, best_idx)
        batch_idx = torch.arange(b)

        selected_true = true_errors[best_idx, batch_idx]
        average_true = true_errors.mean(dim=0)
        gain = (average_true - selected_true) / average_true.clamp_min(1e-8)

        rollout[:, step + 1] = selected
        selected_indices[:, step] = best_idx
        selected_scores[:, step] = score_tensor[best_idx, batch_idx]
        oracle_hit[:, step] = best_idx == oracle_idx
        local_gain[:, step] = gain
        current = selected

    return {
        "rollout": rollout,
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
        "oracle_hit": oracle_hit,
        "local_gain": local_gain,
    }


@torch.inference_mode()
def run_strong_greedy_cyclic(
    model: FlowFormerInference,
    real: torch.Tensor,
    k: int,
    num_transitions: int,
    lookahead: int,
    backward_repeats: int,
    num_ode_steps: int,
    c_forward: torch.Tensor,
    c_backward: torch.Tensor,
    eps: float,
) -> Dict[str, torch.Tensor]:
    b, _, c, h, w = real.shape
    rollout = torch.empty(
        b, num_transitions + 1, c, h, w, dtype=real.dtype
    )
    rollout[:, 0] = real[:, 0]

    selected_indices = torch.empty(b, num_transitions, dtype=torch.long)
    selected_normalized_scores = torch.empty(b, num_transitions)
    selected_raw_scores = torch.empty(b, num_transitions)
    selected_step_energy = torch.empty(b, num_transitions)
    oracle_hit = torch.empty(b, num_transitions, dtype=torch.bool)
    local_gain = torch.empty(b, num_transitions)

    current = rollout[:, 0]

    for step in range(num_transitions):
        effective_lookahead = min(
            lookahead,
            num_transitions - step,
        )

        print(
            f"  strong greedy cyclic {step + 1}/{num_transitions} "
            f"(lookahead={effective_lookahead})",
            flush=True,
        )

        candidate_next_states = []
        candidate_normalized_scores = []
        candidate_raw_scores = []
        candidate_step_energies = []

        for candidate_idx in range(k):
            first_next = sample_one_step(
                model=model,
                x_cond=current,
                num_ode_steps=num_ode_steps,
                direction=c_forward,
            )

            if effective_lookahead == 1:
                forward_path = torch.stack([current, first_next], dim=1)
            else:
                remainder = rollout_direction(
                    model=model,
                    x0=first_next,
                    transitions=effective_lookahead - 1,
                    num_ode_steps=num_ode_steps,
                    direction=c_forward,
                )
                forward_path = torch.cat(
                    [current.unsqueeze(1), remainder],
                    dim=1,
                )

            backward_scores = []
            backward_raw_scores = []
            backward_step_energies = []

            for _ in range(backward_repeats):
                backward_path = rollout_direction(
                    model=model,
                    x0=forward_path[:, -1],
                    transitions=effective_lookahead,
                    num_ode_steps=num_ode_steps,
                    direction=c_backward,
                )

                norm_score, raw_score, step_energy = (
                    normalized_path_cycle_score(
                        forward_path=forward_path,
                        backward_path=backward_path,
                        eps=eps,
                    )
                )

                backward_scores.append(norm_score)
                backward_raw_scores.append(raw_score)
                backward_step_energies.append(step_energy)

            candidate_next_states.append(first_next)
            candidate_normalized_scores.append(
                torch.stack(backward_scores, dim=0).mean(dim=0)
            )
            candidate_raw_scores.append(
                torch.stack(backward_raw_scores, dim=0).mean(dim=0)
            )
            candidate_step_energies.append(
                torch.stack(backward_step_energies, dim=0).mean(dim=0)
            )

        next_tensor = torch.stack(candidate_next_states, dim=0)
        normalized_tensor = torch.stack(
            candidate_normalized_scores,
            dim=0,
        )
        raw_tensor = torch.stack(candidate_raw_scores, dim=0)
        energy_tensor = torch.stack(candidate_step_energies, dim=0)

        true_next = real[:, step + 1]
        true_errors = ((next_tensor - true_next.unsqueeze(0)) ** 2).mean(
            dim=(2, 3, 4)
        )

        best_idx = torch.argmin(normalized_tensor, dim=0)
        oracle_idx = torch.argmin(true_errors, dim=0)
        selected = gather_candidates(next_tensor, best_idx)
        batch_idx = torch.arange(b)

        selected_true = true_errors[best_idx, batch_idx]
        average_true = true_errors.mean(dim=0)
        gain = (average_true - selected_true) / average_true.clamp_min(1e-8)

        rollout[:, step + 1] = selected
        selected_indices[:, step] = best_idx
        selected_normalized_scores[:, step] = normalized_tensor[
            best_idx, batch_idx
        ]
        selected_raw_scores[:, step] = raw_tensor[best_idx, batch_idx]
        selected_step_energy[:, step] = energy_tensor[best_idx, batch_idx]
        oracle_hit[:, step] = best_idx == oracle_idx
        local_gain[:, step] = gain
        current = selected

    return {
        "rollout": rollout,
        "selected_indices": selected_indices,
        "selected_normalized_scores": selected_normalized_scores,
        "selected_raw_scores": selected_raw_scores,
        "selected_step_energy": selected_step_energy,
        "oracle_hit": oracle_hit,
        "local_gain": local_gain,
    }


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    per_time = ((prediction - target) ** 2).mean(dim=(2, 3, 4))
    midpoint = (prediction.shape[1] - 1) // 2

    return {
        "tau0": per_time[:, 0],
        "tau_mid": per_time[:, midpoint],
        "tau_final": per_time[:, -1],
        "rollout_mean": per_time.mean(dim=1),
    }


def aggregate(values: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def summarize_rows(rows: List[dict]) -> Dict[str, dict]:
    methods = sorted({row["method"] for row in rows})
    summary = {}

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}

        for metric in ["tau0", "tau_mid", "tau_final", "rollout_mean"]:
            values = torch.tensor(
                [row[metric] for row in method_rows],
                dtype=torch.float64,
            )
            summary[method][metric] = aggregate(values)

        if "oracle_hit_rate" in method_rows[0]:
            hit_values = torch.tensor(
                [row["oracle_hit_rate"] for row in method_rows],
                dtype=torch.float64,
            )
            gain_values = torch.tensor(
                [row["mean_local_gain"] for row in method_rows],
                dtype=torch.float64,
            )
            summary[method]["oracle_hit_rate"] = aggregate(hit_values)
            summary[method]["mean_local_gain"] = aggregate(gain_values)

    return summary


def print_summary(summary: Dict[str, dict]) -> None:
    print("\n" + "=" * 88)
    print("DATASET-LEVEL SUMMARY")
    print("=" * 88)

    for method, values in summary.items():
        print(f"\n{method}")
        for metric in ["tau0", "tau_mid", "tau_final", "rollout_mean"]:
            item = values[metric]
            print(
                f"  {metric:14s}: "
                f"{item['mean']:.8g} +/- {item['std']:.8g}"
            )

        if "oracle_hit_rate" in values:
            print(
                "  oracle_hit_rate: "
                f"{values['oracle_hit_rate']['mean']:.4f} +/- "
                f"{values['oracle_hit_rate']['std']:.4f}"
            )
            print(
                "  mean_local_gain: "
                f"{100.0 * values['mean_local_gain']['mean']:.3f}% +/- "
                f"{100.0 * values['mean_local_gain']['std']:.3f}%"
            )


def write_txt(
    path: Path,
    model_tag: str,
    args: argparse.Namespace,
    summary: Dict[str, dict],
) -> None:
    with path.open("w") as handle:
        handle.write(f"Model: {model_tag}\n")
        handle.write(f"K: {args.K}\n")
        handle.write(f"Transitions: {args.num_transitions}\n")
        handle.write(f"Lookahead: {args.lookahead}\n")
        handle.write(f"Backward repeats: {args.backward_repeats}\n")
        handle.write(f"ODE steps: {args.num_ode_steps}\n\n")

        for method, values in summary.items():
            handle.write(f"{method}\n")
            for metric in ["tau0", "tau_mid", "tau_final", "rollout_mean"]:
                item = values[metric]
                handle.write(
                    f"{metric}: {item['mean']:.8g} +/- "
                    f"{item['std']:.8g}\n"
                )

            if "oracle_hit_rate" in values:
                handle.write(
                    "oracle_hit_rate: "
                    f"{values['oracle_hit_rate']['mean']:.8g} +/- "
                    f"{values['oracle_hit_rate']['std']:.8g}\n"
                )
                handle.write(
                    "mean_local_gain: "
                    f"{values['mean_local_gain']['mean']:.8g} +/- "
                    f"{values['mean_local_gain']['std']:.8g}\n"
                )

            handle.write("\nLaTeX row:\n")
            handle.write(
                f"& {method.replace('_', ' ')}, $K={args.K}$\n"
                f"& {values['tau0']['mean']:.6f} $\\pm$ "
                f"{values['tau0']['std']:.6f}\n"
                f"& {values['tau_mid']['mean']:.6f} $\\pm$ "
                f"{values['tau_mid']['std']:.6f}\n"
                f"& {values['tau_final']['mean']:.6f} $\\pm$ "
                f"{values['tau_final']['std']:.6f}\n"
                f"& {values['rollout_mean']['mean']:.6f} $\\pm$ "
                f"{values['rollout_mean']['std']:.6f} \\\\\n\n"
            )


def main() -> None:
    args = parse_args()

    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.num_transitions < 1:
        raise ValueError("--num_transitions must be >= 1")
    if args.lookahead < 1:
        raise ValueError("--lookahead must be >= 1")
    if args.backward_repeats < 1:
        raise ValueError("--backward_repeats must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["args"]
    model_size = config["ms"]
    dataset_name = config["ds"]

    videos = np.load(dataset_path, mmap_mode="r")
    if videos.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W], got {videos.shape}")

    n_total, t_total, _, height, width = videos.shape
    required_states = args.num_transitions + 1

    if t_total < required_states:
        raise ValueError(
            f"Dataset has {t_total} states, but {required_states} are required."
        )

    if args.all_trajs:
        trajectory_indices = list(range(n_total))
    elif args.traj_indices is None:
        trajectory_indices = [0, n_total // 2, n_total - 1]
    else:
        trajectory_indices = args.traj_indices

    for index in trajectory_indices:
        if index < 0 or index >= n_total:
            raise IndexError(
                f"Trajectory {index} is outside [0, {n_total - 1}]"
            )

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)
    c_backward = torch.tensor([0.0, 1.0], dtype=torch.float32)

    print("=" * 88)
    print("Strong short-horizon greedy cyclic evaluation")
    print("=" * 88)
    print("Checkpoint:", checkpoint_path)
    print("Dataset:", dataset_path)
    print("Model size:", model_size)
    print("Dataset name:", dataset_name)
    print("Device:", device)
    print("Trajectories:", len(trajectory_indices))
    print("Batch size:", args.batch_size)
    print("K:", args.K)
    print("Transitions:", args.num_transitions)
    print("Lookahead:", args.lookahead)
    print("Backward repeats:", args.backward_repeats)
    print("ODE steps:", args.num_ode_steps)
    print("=" * 88)

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=str(checkpoint_path),
        img_size=(height, width),
        device=device,
    )

    rows: List[dict] = []
    selection_details = {
        "stepwise_oracle": [],
        "raw_greedy_cyclic": [],
        "strong_greedy_cyclic": [],
    }

    for start in range(0, len(trajectory_indices), args.batch_size):
        batch_indices = trajectory_indices[start : start + args.batch_size]

        print(
            f"\nBatch trajectories: {batch_indices}",
            flush=True,
        )

        real = torch.from_numpy(
            np.asarray(videos[batch_indices, :required_states])
        ).float()

        oracle_result = run_stepwise_oracle(
            model=model,
            real=real,
            k=args.K,
            num_transitions=args.num_transitions,
            num_ode_steps=args.num_ode_steps,
            c_forward=c_forward,
        )

        raw_result = run_raw_greedy_cyclic(
            model=model,
            real=real,
            k=args.K,
            num_transitions=args.num_transitions,
            num_ode_steps=args.num_ode_steps,
            c_forward=c_forward,
            c_backward=c_backward,
        )

        strong_result = run_strong_greedy_cyclic(
            model=model,
            real=real,
            k=args.K,
            num_transitions=args.num_transitions,
            lookahead=args.lookahead,
            backward_repeats=args.backward_repeats,
            num_ode_steps=args.num_ode_steps,
            c_forward=c_forward,
            c_backward=c_backward,
            eps=args.eps,
        )

        results = {
            "stepwise_oracle": oracle_result,
            "raw_greedy_cyclic": raw_result,
            "strong_greedy_cyclic": strong_result,
        }

        for method_name, result in results.items():
            metrics = compute_metrics(result["rollout"], real)

            for local_idx, trajectory_index in enumerate(batch_indices):
                row = {
                    "model": f"flowformer_{model_size}_{dataset_name}",
                    "dataset": dataset_name,
                    "method": method_name,
                    "trajectory_index": trajectory_index,
                    "tau0": float(metrics["tau0"][local_idx].item()),
                    "tau_mid": float(metrics["tau_mid"][local_idx].item()),
                    "tau_final": float(metrics["tau_final"][local_idx].item()),
                    "rollout_mean": float(
                        metrics["rollout_mean"][local_idx].item()
                    ),
                    "K": args.K,
                    "num_transitions": args.num_transitions,
                    "lookahead": args.lookahead,
                    "backward_repeats": args.backward_repeats,
                }

                if "oracle_hit" in result:
                    row["oracle_hit_rate"] = float(
                        result["oracle_hit"][local_idx].float().mean().item()
                    )
                    row["mean_local_gain"] = float(
                        result["local_gain"][local_idx].mean().item()
                    )

                rows.append(row)

            detail_copy = {
                key: value.detach().cpu()
                for key, value in result.items()
                if torch.is_tensor(value) and key != "rollout"
            }
            detail_copy["trajectory_indices"] = batch_indices
            selection_details[method_name].append(detail_copy)

    summary = summarize_rows(rows)
    print_summary(summary)

    model_tag = f"flowformer_{model_size}_{dataset_name}"
    suffix = "all_trajs" if args.all_trajs else "selected_trajs"
    basename = (
        f"strong_greedy_{model_tag}_K{args.K}_"
        f"L{args.lookahead}_R{args.backward_repeats}_"
        f"T{args.num_transitions}_{suffix}"
    )

    csv_path = output_dir / f"{basename}.csv"
    pt_path = output_dir / f"{basename}.pt"
    txt_path = output_dir / f"{basename}.txt"

    fieldnames = [
        "model",
        "dataset",
        "method",
        "trajectory_index",
        "tau0",
        "tau_mid",
        "tau_final",
        "rollout_mean",
        "K",
        "num_transitions",
        "lookahead",
        "backward_repeats",
        "oracle_hit_rate",
        "mean_local_gain",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_args": config,
            "dataset": str(dataset_path),
            "trajectory_indices": trajectory_indices,
            "K": args.K,
            "num_transitions": args.num_transitions,
            "lookahead": args.lookahead,
            "backward_repeats": args.backward_repeats,
            "num_ode_steps": args.num_ode_steps,
            "rows": rows,
            "summary": summary,
            "selection_details": selection_details,
        },
        pt_path,
    )

    write_txt(
        path=txt_path,
        model_tag=model_tag,
        args=args,
        summary=summary,
    )

    print("\nSaved:")
    print(csv_path)
    print(pt_path)
    print(txt_path)


if __name__ == "__main__":
    main()
