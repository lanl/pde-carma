#!/usr/bin/env python3
"""
Corrected adaptive short-horizon cyclic lookahead for bidirectional FlowFormer.

Key correction:
- Predictive variance chooses ONE COMMON horizon per trajectory at each step.
- All K candidates for that trajectory are evaluated with the same horizon.
- Candidate ranking is based primarily on normalized cyclic consistency.
- Optional small variance penalty can be added with --variance_weight.

At each transition:
  1. Generate K candidate next states.
  2. Estimate predictive variance for each candidate from M forward samples.
  3. Compute step-level uncertainty by averaging across candidates.
  4. Choose one common lookahead horizon for that trajectory:
       low uncertainty    -> lookahead_min
       medium uncertainty -> midpoint horizon
       high uncertainty   -> lookahead_max
  5. Evaluate every candidate using that same horizon.
  6. Score each candidate with normalized short-path cycle error.
  7. Optionally add a small normalized predictive-variance penalty.
  8. Select the lowest-scoring candidate and continue greedily.

Ground truth is used only for evaluation diagnostics.
"""

import argparse
import csv
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_ode_steps", type=int, default=5)
    parser.add_argument("--num_transitions", type=int, default=20)
    parser.add_argument("--K", type=int, default=4)

    parser.add_argument(
        "--variance_samples",
        type=int,
        default=4,
        help="Forward samples used to estimate predictive variance.",
    )
    parser.add_argument("--lookahead_min", type=int, default=2)
    parser.add_argument("--lookahead_max", type=int, default=6)
    parser.add_argument("--backward_repeats", type=int, default=2)

    parser.add_argument(
        "--variance_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of normalized predictive variance in candidate ranking. "
            "Use 0.0 for cycle-only ranking; try 0.1 as a secondary experiment."
        ),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,
        help=(
            "Before enough uncertainty history exists, use the midpoint horizon."
        ),
    )
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--out_dir",
        default="results/corrected_adaptive_cyclic",
    )
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
    batch_size, channels, height, width = x0.shape

    trajectory = torch.empty(
        batch_size,
        transitions + 1,
        channels,
        height,
        width,
        dtype=x0.dtype,
    )
    trajectory[:, 0] = x0.detach().cpu()

    current = trajectory[:, 0]

    for step in range(transitions):
        current = sample_one_step(
            model=model,
            x_cond=current,
            num_ode_steps=num_ode_steps,
            direction=direction,
        )
        trajectory[:, step + 1] = current

    return trajectory


def gather_candidates(
    candidates: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """
    candidates:       [K, B, ...]
    selected_indices: [B]
    """
    batch_indices = torch.arange(candidates.shape[1])
    return candidates[selected_indices, batch_indices]


@torch.inference_mode()
def estimate_predictive_variance(
    model: FlowFormerInference,
    candidate: torch.Tensor,
    num_samples: int,
    num_ode_steps: int,
    c_forward: torch.Tensor,
) -> torch.Tensor:
    """
    Draw multiple one-step futures from one candidate.

    Returns scalar predictive variance per trajectory: [B].
    """
    futures = []

    for _ in range(num_samples):
        futures.append(
            sample_one_step(
                model=model,
                x_cond=candidate,
                num_ode_steps=num_ode_steps,
                direction=c_forward,
            )
        )

    futures_tensor = torch.stack(futures, dim=0)  # [M,B,C,H,W]

    return futures_tensor.var(
        dim=0,
        unbiased=False,
    ).mean(dim=(1, 2, 3))


def minmax_normalize(
    values: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Normalize across candidates independently for each trajectory.

    values: [K,B]
    """
    minimum = values.min(dim=0).values
    maximum = values.max(dim=0).values

    return (
        values - minimum.unsqueeze(0)
    ) / (
        maximum - minimum
    ).clamp_min(eps).unsqueeze(0)


def normalized_path_cycle_score(
    forward_path: torch.Tensor,
    backward_path: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    forward_path:  [B,L+1,C,H,W]
    backward_path: [B,L+1,C,H,W]

    Returns:
      normalized cycle score [B]
      raw path cycle score   [B]
      forward path energy    [B]
    """
    reversed_forward = torch.flip(forward_path, dims=[1])

    residual = (
        (backward_path - reversed_forward) ** 2
    ).mean(dim=(2, 3, 4))

    # Index 0 is the shared terminal endpoint.
    raw_cycle_score = residual[:, 1:].mean(dim=1)

    forward_deltas = forward_path[:, 1:] - forward_path[:, :-1]
    path_energy = (
        forward_deltas ** 2
    ).mean(dim=(2, 3, 4)).mean(dim=1)

    normalized_score = (
        raw_cycle_score / path_energy.clamp_min(eps)
    )

    return normalized_score, raw_cycle_score, path_energy


def choose_common_horizon(
    step_uncertainty: torch.Tensor,
    uncertainty_history: List[torch.Tensor],
    lookahead_min: int,
    lookahead_max: int,
    remaining_steps: int,
    warmup_steps: int,
) -> torch.Tensor:
    """
    Choose ONE horizon per trajectory for the current transition.

    step_uncertainty: [B]
    returns:          [B]
    """
    middle_horizon = int(
        round((lookahead_min + lookahead_max) / 2)
    )

    if len(uncertainty_history) < warmup_steps:
        horizon = torch.full(
            step_uncertainty.shape,
            middle_horizon,
            dtype=torch.long,
        )
    else:
        historical = torch.stack(
            uncertainty_history,
            dim=0,
        )  # [steps,B]

        low_threshold = torch.quantile(
            historical,
            0.33,
            dim=0,
        )
        high_threshold = torch.quantile(
            historical,
            0.67,
            dim=0,
        )

        horizon = torch.full(
            step_uncertainty.shape,
            lookahead_min,
            dtype=torch.long,
        )

        medium_mask = step_uncertainty > low_threshold
        high_mask = step_uncertainty > high_threshold

        horizon[medium_mask] = middle_horizon
        horizon[high_mask] = lookahead_max

    return horizon.clamp(
        min=1,
        max=remaining_steps,
    )


@torch.inference_mode()
def run_corrected_adaptive_selector(
    model: FlowFormerInference,
    real: torch.Tensor,
    args: argparse.Namespace,
    c_forward: torch.Tensor,
    c_backward: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    batch_size, _, channels, height, width = real.shape

    rollout = torch.empty(
        batch_size,
        args.num_transitions + 1,
        channels,
        height,
        width,
        dtype=real.dtype,
    )
    rollout[:, 0] = real[:, 0]

    selected_indices = torch.empty(
        batch_size,
        args.num_transitions,
        dtype=torch.long,
    )
    selected_horizons = torch.empty(
        batch_size,
        args.num_transitions,
        dtype=torch.long,
    )
    selected_cycle_scores = torch.empty(
        batch_size,
        args.num_transitions,
    )
    selected_variance_scores = torch.empty(
        batch_size,
        args.num_transitions,
    )
    selected_combined_scores = torch.empty(
        batch_size,
        args.num_transitions,
    )
    oracle_hit = torch.empty(
        batch_size,
        args.num_transitions,
        dtype=torch.bool,
    )
    local_gain = torch.empty(
        batch_size,
        args.num_transitions,
    )

    uncertainty_history: List[torch.Tensor] = []
    current = rollout[:, 0]

    for step in range(args.num_transitions):
        remaining_steps = args.num_transitions - step

        print(
            f"  corrected adaptive greedy "
            f"{step + 1}/{args.num_transitions}",
            flush=True,
        )

        next_candidates = []
        candidate_variances = []

        # Generate K next-state candidates and estimate each candidate's
        # predictive uncertainty.
        for _ in range(args.K):
            candidate = sample_one_step(
                model=model,
                x_cond=current,
                num_ode_steps=args.num_ode_steps,
                direction=c_forward,
            )

            variance = estimate_predictive_variance(
                model=model,
                candidate=candidate,
                num_samples=args.variance_samples,
                num_ode_steps=args.num_ode_steps,
                c_forward=c_forward,
            )

            next_candidates.append(candidate)
            candidate_variances.append(variance)

        next_tensor = torch.stack(
            next_candidates,
            dim=0,
        )  # [K,B,C,H,W]

        variance_tensor = torch.stack(
            candidate_variances,
            dim=0,
        )  # [K,B]

        # One uncertainty value and one common horizon per trajectory.
        step_uncertainty = variance_tensor.mean(dim=0)

        common_horizon = choose_common_horizon(
            step_uncertainty=step_uncertainty,
            uncertainty_history=uncertainty_history,
            lookahead_min=args.lookahead_min,
            lookahead_max=args.lookahead_max,
            remaining_steps=remaining_steps,
            warmup_steps=args.warmup_steps,
        )

        # Add the current uncertainty after selecting the threshold from
        # previous steps, preventing current-step leakage into its own bins.
        uncertainty_history.append(
            step_uncertainty.detach().clone()
        )

        candidate_cycle_scores = []
        candidate_raw_scores = []
        candidate_path_energies = []

        # Every candidate for the same trajectory is scored with the same
        # horizon. Batch elements may still have different horizons.
        for candidate_index in range(args.K):
            per_batch_cycle = []
            per_batch_raw = []
            per_batch_energy = []

            for batch_index in range(batch_size):
                horizon = int(
                    common_horizon[batch_index].item()
                )

                current_one = current[
                    batch_index : batch_index + 1
                ]
                candidate_one = next_tensor[
                    candidate_index,
                    batch_index : batch_index + 1,
                ]

                if horizon == 1:
                    forward_path = torch.stack(
                        [current_one, candidate_one],
                        dim=1,
                    )
                else:
                    remainder_path = rollout_direction(
                        model=model,
                        x0=candidate_one,
                        transitions=horizon - 1,
                        num_ode_steps=args.num_ode_steps,
                        direction=c_forward,
                    )

                    forward_path = torch.cat(
                        [
                            current_one.unsqueeze(1),
                            remainder_path,
                        ],
                        dim=1,
                    )

                repeated_cycle_scores = []
                repeated_raw_scores = []
                repeated_energies = []

                for _ in range(args.backward_repeats):
                    backward_path = rollout_direction(
                        model=model,
                        x0=forward_path[:, -1],
                        transitions=horizon,
                        num_ode_steps=args.num_ode_steps,
                        direction=c_backward,
                    )

                    normalized_score, raw_score, energy = (
                        normalized_path_cycle_score(
                            forward_path=forward_path,
                            backward_path=backward_path,
                            eps=args.eps,
                        )
                    )

                    repeated_cycle_scores.append(
                        normalized_score
                    )
                    repeated_raw_scores.append(raw_score)
                    repeated_energies.append(energy)

                per_batch_cycle.append(
                    torch.stack(
                        repeated_cycle_scores,
                        dim=0,
                    ).mean()
                )
                per_batch_raw.append(
                    torch.stack(
                        repeated_raw_scores,
                        dim=0,
                    ).mean()
                )
                per_batch_energy.append(
                    torch.stack(
                        repeated_energies,
                        dim=0,
                    ).mean()
                )

            candidate_cycle_scores.append(
                torch.stack(per_batch_cycle)
            )
            candidate_raw_scores.append(
                torch.stack(per_batch_raw)
            )
            candidate_path_energies.append(
                torch.stack(per_batch_energy)
            )

        cycle_tensor = torch.stack(
            candidate_cycle_scores,
            dim=0,
        )  # [K,B]

        raw_tensor = torch.stack(
            candidate_raw_scores,
            dim=0,
        )
        energy_tensor = torch.stack(
            candidate_path_energies,
            dim=0,
        )

        cycle_rank = minmax_normalize(
            cycle_tensor,
            args.eps,
        )
        variance_rank = minmax_normalize(
            variance_tensor,
            args.eps,
        )

        combined_score = (
            (1.0 - args.variance_weight) * cycle_rank
            + args.variance_weight * variance_rank
        )

        best_index = torch.argmin(
            combined_score,
            dim=0,
        )
        selected = gather_candidates(
            next_tensor,
            best_index,
        )
        batch_indices = torch.arange(batch_size)

        # Evaluation-only diagnostics.
        true_next = real[:, step + 1]
        true_errors = (
            (next_tensor - true_next.unsqueeze(0)) ** 2
        ).mean(dim=(2, 3, 4))

        oracle_index = torch.argmin(
            true_errors,
            dim=0,
        )

        selected_true_error = true_errors[
            best_index,
            batch_indices,
        ]
        average_true_error = true_errors.mean(dim=0)

        gain = (
            average_true_error - selected_true_error
        ) / average_true_error.clamp_min(args.eps)

        rollout[:, step + 1] = selected
        selected_indices[:, step] = best_index
        selected_horizons[:, step] = common_horizon
        selected_cycle_scores[:, step] = cycle_tensor[
            best_index,
            batch_indices,
        ]
        selected_variance_scores[:, step] = variance_tensor[
            best_index,
            batch_indices,
        ]
        selected_combined_scores[:, step] = combined_score[
            best_index,
            batch_indices,
        ]
        oracle_hit[:, step] = best_index == oracle_index
        local_gain[:, step] = gain

        print(
            "    horizon="
            f"{common_horizon.float().mean().item():.2f}, "
            "oracle hit="
            f"{oracle_hit[:, step].float().mean().item():.3f}, "
            "gain="
            f"{100.0 * local_gain[:, step].mean().item():.2f}%",
            flush=True,
        )

        current = selected

    return {
        "rollout": rollout,
        "selected_indices": selected_indices,
        "selected_horizons": selected_horizons,
        "selected_cycle_scores": selected_cycle_scores,
        "selected_variance_scores": selected_variance_scores,
        "selected_combined_scores": selected_combined_scores,
        "oracle_hit": oracle_hit,
        "local_gain": local_gain,
    }


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    per_time = (
        (prediction - target) ** 2
    ).mean(dim=(2, 3, 4))

    midpoint = (
        prediction.shape[1] - 1
    ) // 2

    return {
        "tau0": per_time[:, 0],
        "tau_mid": per_time[:, midpoint],
        "tau_final": per_time[:, -1],
        "rollout_mean": per_time.mean(dim=1),
    }


def aggregate(
    values: torch.Tensor,
) -> Dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std": float(
            values.std(unbiased=False).item()
        ),
    }


def main() -> None:
    args = parse_args()

    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.variance_samples < 2:
        raise ValueError(
            "--variance_samples must be >= 2"
        )
    if args.lookahead_min < 1:
        raise ValueError(
            "--lookahead_min must be >= 1"
        )
    if args.lookahead_max < args.lookahead_min:
        raise ValueError(
            "--lookahead_max must be >= --lookahead_min"
        )
    if args.backward_repeats < 1:
        raise ValueError(
            "--backward_repeats must be >= 1"
        )
    if not 0.0 <= args.variance_weight <= 1.0:
        raise ValueError(
            "--variance_weight must be in [0,1]"
        )
    if args.batch_size < 1:
        raise ValueError(
            "--batch_size must be >= 1"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    output_dir = Path(
        args.out_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    config = checkpoint["args"]

    model_size = config["ms"]
    dataset_name = config["ds"]

    videos = np.load(
        dataset_path,
        mmap_mode="r",
    )

    if videos.ndim != 5:
        raise ValueError(
            f"Expected [N,T,C,H,W], got {videos.shape}"
        )

    n_total, t_total, _, height, width = videos.shape
    required_states = args.num_transitions + 1

    if t_total < required_states:
        raise ValueError(
            f"Dataset has {t_total} states; "
            f"{required_states} are required."
        )

    if args.all_trajs:
        trajectory_indices = list(
            range(n_total)
        )
    elif args.traj_indices is None:
        trajectory_indices = [
            0,
            n_total // 2,
            n_total - 1,
        ]
    else:
        trajectory_indices = args.traj_indices

    for index in trajectory_indices:
        if index < 0 or index >= n_total:
            raise IndexError(
                f"Trajectory {index} is outside "
                f"[0, {n_total - 1}]"
            )

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    c_forward = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )
    c_backward = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float32,
    )

    print("=" * 88)
    print("Corrected adaptive cyclic lookahead")
    print("=" * 88)
    print("Checkpoint:", checkpoint_path)
    print("Dataset:", dataset_path)
    print("Device:", device)
    print("Trajectories:", len(trajectory_indices))
    print("K:", args.K)
    print("Variance samples:", args.variance_samples)
    print(
        "Lookahead range:",
        f"{args.lookahead_min}..{args.lookahead_max}",
    )
    print("Backward repeats:", args.backward_repeats)
    print("Variance weight:", args.variance_weight)
    print("Warmup steps:", args.warmup_steps)
    print("=" * 88)

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=str(checkpoint_path),
        img_size=(height, width),
        device=device,
    )

    rows: List[dict] = []
    selection_details: List[dict] = []

    for start in range(
        0,
        len(trajectory_indices),
        args.batch_size,
    ):
        batch_indices = trajectory_indices[
            start : start + args.batch_size
        ]

        print(
            f"\nBatch trajectories: {batch_indices}",
            flush=True,
        )

        real = torch.from_numpy(
            np.asarray(
                videos[
                    batch_indices,
                    :required_states,
                ]
            )
        ).float()

        result = run_corrected_adaptive_selector(
            model=model,
            real=real,
            args=args,
            c_forward=c_forward,
            c_backward=c_backward,
        )

        metrics = compute_metrics(
            result["rollout"],
            real,
        )

        for local_index, trajectory_index in enumerate(
            batch_indices
        ):
            rows.append(
                {
                    "model": (
                        f"flowformer_{model_size}_"
                        f"{dataset_name}"
                    ),
                    "dataset": dataset_name,
                    "method": (
                        "corrected_adaptive_cyclic"
                    ),
                    "trajectory_index": trajectory_index,
                    "tau0": float(
                        metrics["tau0"][
                            local_index
                        ].item()
                    ),
                    "tau_mid": float(
                        metrics["tau_mid"][
                            local_index
                        ].item()
                    ),
                    "tau_final": float(
                        metrics["tau_final"][
                            local_index
                        ].item()
                    ),
                    "rollout_mean": float(
                        metrics["rollout_mean"][
                            local_index
                        ].item()
                    ),
                    "mean_selected_horizon": float(
                        result["selected_horizons"][
                            local_index
                        ].float().mean().item()
                    ),
                    "oracle_hit_rate": float(
                        result["oracle_hit"][
                            local_index
                        ].float().mean().item()
                    ),
                    "mean_local_gain": float(
                        result["local_gain"][
                            local_index
                        ].mean().item()
                    ),
                    "K": args.K,
                    "variance_samples": (
                        args.variance_samples
                    ),
                    "lookahead_min": (
                        args.lookahead_min
                    ),
                    "lookahead_max": (
                        args.lookahead_max
                    ),
                    "backward_repeats": (
                        args.backward_repeats
                    ),
                    "variance_weight": (
                        args.variance_weight
                    ),
                }
            )

        selection_details.append(
            {
                "trajectory_indices": batch_indices,
                **{
                    key: value.detach().cpu()
                    for key, value in result.items()
                    if (
                        torch.is_tensor(value)
                        and key != "rollout"
                    )
                },
            }
        )

    metric_names = [
        "tau0",
        "tau_mid",
        "tau_final",
        "rollout_mean",
        "mean_selected_horizon",
        "oracle_hit_rate",
        "mean_local_gain",
    ]

    summary = {}

    for metric_name in metric_names:
        values = torch.tensor(
            [
                row[metric_name]
                for row in rows
            ],
            dtype=torch.float64,
        )
        summary[metric_name] = aggregate(values)

    print("\n" + "=" * 88)
    print("DATASET-LEVEL SUMMARY")
    print("=" * 88)

    for metric_name in metric_names:
        item = summary[metric_name]

        if metric_name == "mean_local_gain":
            print(
                f"{metric_name:24s}: "
                f"{100.0 * item['mean']:.8g}% +/- "
                f"{100.0 * item['std']:.8g}%"
            )
        else:
            print(
                f"{metric_name:24s}: "
                f"{item['mean']:.8g} +/- "
                f"{item['std']:.8g}"
            )

    tag = (
        f"flowformer_{model_size}_{dataset_name}"
    )
    suffix = (
        "all_trajs"
        if args.all_trajs
        else "selected_trajs"
    )
    basename = (
        f"corrected_adaptive_{tag}_"
        f"K{args.K}_"
        f"M{args.variance_samples}_"
        f"L{args.lookahead_min}-"
        f"{args.lookahead_max}_"
        f"R{args.backward_repeats}_"
        f"V{args.variance_weight:.2f}_"
        f"T{args.num_transitions}_"
        f"{suffix}"
    )

    csv_path = output_dir / (
        basename + ".csv"
    )
    pt_path = output_dir / (
        basename + ".pt"
    )
    txt_path = output_dir / (
        basename + ".txt"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get(
                "epoch"
            ),
            "checkpoint_args": config,
            "dataset": str(dataset_path),
            "trajectory_indices": (
                trajectory_indices
            ),
            "args": vars(args),
            "rows": rows,
            "summary": summary,
            "selection_details": (
                selection_details
            ),
        },
        pt_path,
    )

    with txt_path.open("w") as handle:
        handle.write(f"Model: {tag}\n")

        for key, value in vars(args).items():
            handle.write(f"{key}: {value}\n")

        handle.write(
            "\nDataset-level summary\n"
        )

        for metric_name in metric_names:
            item = summary[metric_name]
            handle.write(
                f"{metric_name}: "
                f"{item['mean']:.8g} +/- "
                f"{item['std']:.8g}\n"
            )

        handle.write("\nLaTeX row\n")
        handle.write(
            f"& Corrected adaptive cyclic, "
            f"$K={args.K}$\n"
            f"& {summary['tau0']['mean']:.6f} "
            f"$\\pm$ {summary['tau0']['std']:.6f}\n"
            f"& {summary['tau_mid']['mean']:.6f} "
            f"$\\pm$ {summary['tau_mid']['std']:.6f}\n"
            f"& {summary['tau_final']['mean']:.6f} "
            f"$\\pm$ {summary['tau_final']['std']:.6f}\n"
            f"& {summary['rollout_mean']['mean']:.6f} "
            f"$\\pm$ "
            f"{summary['rollout_mean']['std']:.6f} "
            f"\\\\\n"
        )

    print("\nSaved:")
    print(csv_path)
    print(pt_path)
    print(txt_path)


if __name__ == "__main__":
    main()
