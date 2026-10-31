import argparse
import csv
import inspect
import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from utils.device_manager import DeviceManager
from utils.my_model import DiTInference


def resolve_path(path):
    """Resolve a path relative to the repository root."""
    if os.path.isfile(path):
        return path

    resolved = os.path.join(PROJECT_ROOT, path)

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Could not find file:\n"
            f"  supplied: {path}\n"
            f"  resolved: {resolved}"
        )

    return resolved


def get_config_value(config, name):
    """Support checkpoint args stored as either dict or Namespace."""
    if isinstance(config, dict):
        return config[name]

    return getattr(config, name)


def extract_final_sample(output, reference):
    """
    Convert the output of DiTInference.sample() to a tensor with shape
    (N, C, H, W).

    This supports common return formats:
      - final tensor: (N, C, H, W)
      - sampling path: (N, S, C, H, W)
      - sampling path: (S, N, C, H, W)
      - list/tuple of sampling states
      - dictionary containing a generated sample
    """
    if isinstance(output, dict):
        preferred_keys = [
            "sample",
            "samples",
            "x",
            "x_final",
            "final",
            "trajectory",
            "path",
        ]

        for key in preferred_keys:
            if key in output:
                return extract_final_sample(output[key], reference)

        raise ValueError(
            "DiT sample returned a dictionary, but no recognized sample "
            f"key was found. Keys: {list(output.keys())}"
        )

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("DiT sample returned an empty list/tuple.")

        return extract_final_sample(output[-1], reference)

    if not torch.is_tensor(output):
        raise TypeError(
            "Unsupported output type from DiT sample: "
            f"{type(output).__name__}"
        )

    if output.ndim == 4:
        final = output

    elif output.ndim == 5:
        batch_size = reference.shape[0]

        if output.shape[0] == batch_size:
            # Expected form: (N, sampling_steps, C, H, W)
            final = output[:, -1]

        elif output.shape[1] == batch_size:
            # Alternative form: (sampling_steps, N, C, H, W)
            final = output[-1]

        else:
            raise ValueError(
                "Could not determine sampling dimension for DiT output "
                f"with shape {tuple(output.shape)}."
            )

    else:
        raise ValueError(
            "Expected DiT output to have 4 or 5 dimensions, but got "
            f"shape {tuple(output.shape)}."
        )

    if tuple(final.shape) != tuple(reference.shape):
        raise ValueError(
            "Generated sample has unexpected shape.\n"
            f"Expected: {tuple(reference.shape)}\n"
            f"Received: {tuple(final.shape)}"
        )

    return final


def sample_dit(
    model,
    x_cond,
    control,
    samp_diff_steps,
    ddim_eta,
):
    """
    Call DiTInference.sample() while adapting to the exact argument names
    supported by the local repository.
    """
    signature = inspect.signature(model.sample)
    supported = set(signature.parameters.keys())

    kwargs = {}

    if "x_cond" in supported:
        kwargs["x_cond"] = x_cond
    elif "cond" in supported:
        kwargs["cond"] = x_cond
    elif "condition" in supported:
        kwargs["condition"] = x_cond
    else:
        raise TypeError(
            "Could not identify the conditioning argument in "
            f"model.sample{signature}"
        )

    if "c" in supported:
        kwargs["c"] = control
    elif "control" in supported:
        kwargs["control"] = control
    elif "direction" in supported:
        kwargs["direction"] = control

    if "samp_diff_steps" in supported:
        kwargs["samp_diff_steps"] = samp_diff_steps
    elif "sampling_steps" in supported:
        kwargs["sampling_steps"] = samp_diff_steps
    elif "num_sampling_steps" in supported:
        kwargs["num_sampling_steps"] = samp_diff_steps
    elif "num_steps" in supported:
        kwargs["num_steps"] = samp_diff_steps

    if "ddim_eta" in supported:
        kwargs["ddim_eta"] = ddim_eta
    elif "eta" in supported:
        kwargs["eta"] = ddim_eta

    output = model.sample(**kwargs)
    return extract_final_sample(output, reference=x_cond)


def rollout_forward(
    model,
    x0,
    T_roll,
    samp_diff_steps,
    ddim_eta,
    c_forward,
):
    """
    Generate a forward autoregressive trajectory.

    x0:
        (N, C, H, W)

    returns:
        (N, T_roll, C, H, W)
    """
    N, C, H, W = x0.shape

    trajectory = torch.zeros(
        N,
        T_roll,
        C,
        H,
        W,
        dtype=torch.float32,
    )

    trajectory[:, 0] = x0.detach().cpu()
    condition = trajectory[:, 0]

    for t in range(1, T_roll):
        x_next = sample_dit(
            model=model,
            x_cond=condition,
            control=c_forward,
            samp_diff_steps=samp_diff_steps,
            ddim_eta=ddim_eta,
        )

        x_next = x_next.detach().cpu()
        trajectory[:, t] = x_next
        condition = x_next

        print(f"  forward step {t}/{T_roll - 1}")

    return trajectory


def rollout_backward(
    model,
    xT,
    T_roll,
    samp_diff_steps,
    ddim_eta,
    c_backward,
):
    """
    Generate a backward autoregressive trajectory.

    The returned trajectory is ordered:

        generated terminal frame,
        generated previous frame,
        ...,
        reconstructed initial frame.
    """
    N, C, H, W = xT.shape

    trajectory = torch.zeros(
        N,
        T_roll,
        C,
        H,
        W,
        dtype=torch.float32,
    )

    trajectory[:, 0] = xT.detach().cpu()
    condition = trajectory[:, 0]

    for t in range(1, T_roll):
        x_previous = sample_dit(
            model=model,
            x_cond=condition,
            control=c_backward,
            samp_diff_steps=samp_diff_steps,
            ddim_eta=ddim_eta,
        )

        x_previous = x_previous.detach().cpu()
        trajectory[:, t] = x_previous
        condition = x_previous

        print(f"  backward step {t}/{T_roll - 1}")

    return trajectory


def rollout_error_per_trajectory(predicted, ground_truth):
    """
    Compute one complete-trajectory MSE per test trajectory.

    predicted, ground_truth:
        (N, T, C, H, W)

    returns:
        (N,)
    """
    return ((predicted - ground_truth) ** 2).mean(
        dim=(1, 2, 3, 4)
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate forward and cyclic Overleaf-table results for "
            "a bidirectional Diffusion Transformer."
        )
    )

    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/vids_test_ce-rm.npy",
    )

    parser.add_argument(
        "--samp_diff_steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max_T_rollout",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--K_forward",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--K_cyclic",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--traj_indices",
        type=int,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--all_trajs",
        action="store_true",
    )

    args = parser.parse_args()

    if args.K_forward <= 0:
        raise ValueError("--K_forward must be positive.")

    if args.K_cyclic <= 0:
        raise ValueError("--K_cyclic must be positive.")

    if args.samp_diff_steps <= 0:
        raise ValueError("--samp_diff_steps must be positive.")

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "args" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain an 'args' configuration."
        )

    config = checkpoint["args"]
    model_size = get_config_value(config, "ms")
    dataset_name = get_config_value(config, "ds")
    dit_arch = (
        config.get("dit_arch", "legacy")
        if isinstance(config, dict)
        else getattr(config, "dit_arch", "legacy")
    )

    devices = DeviceManager.list_devices()

    if args.device_idx >= len(devices):
        raise IndexError(
            f"device_idx={args.device_idx}, but only "
            f"{len(devices)} device(s) were found."
        )

    device = devices[args.device_idx]

    videos_test = torch.from_numpy(
        np.load(dataset_path)
    ).float()

    if videos_test.ndim != 5:
        raise ValueError(
            "Expected test data with shape (N,T,C,H,W), but got "
            f"{tuple(videos_test.shape)}."
        )

    N_total, T_total, C, H, W = videos_test.shape

    if args.all_trajs:
        trajectory_indices = list(range(N_total))

    elif args.traj_indices is None:
        trajectory_indices = [
            0,
            N_total // 2,
            N_total - 1,
        ]

    else:
        trajectory_indices = args.traj_indices

    for index in trajectory_indices:
        if index < 0 or index >= N_total:
            raise IndexError(
                f"Trajectory index {index} is outside the valid range "
                f"[0, {N_total - 1}]."
            )

    selected_videos = videos_test[trajectory_indices]

    T_roll = min(
        args.max_T_rollout,
        selected_videos.shape[1],
    )

    ground_truth_forward = (
        selected_videos[:, :T_roll]
        .detach()
        .cpu()
    )

    c_forward = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    c_backward = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float32,
    )

    print("===================================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Model size: {model_size}")
    print(f"Dataset name: {dataset_name}")
    print(f"Test-set shape: {tuple(videos_test.shape)}")
    print(f"Selected trajectories: {trajectory_indices}")
    print(f"T_roll: {T_roll}")
    print(f"K_forward: {args.K_forward}")
    print(f"K_cyclic: {args.K_cyclic}")
    print(f"samp_diff_steps: {args.samp_diff_steps}")
    print(f"ddim_eta: {args.ddim_eta}")
    print(f"Device: {device}")
    print("===================================")

    model = DiTInference.from_pretrained(
        checkpoint_path=checkpoint_path,
        img_size=(H, W),
        device=device,
    )

    # -------------------------------------------------------
    # Forward stochastic rollouts
    # -------------------------------------------------------
    forward_errors = []

    for candidate_index in range(args.K_forward):
        print(
            f"\nForward candidate "
            f"{candidate_index + 1}/{args.K_forward}"
        )

        generated_forward = rollout_forward(
            model=model,
            x0=ground_truth_forward[:, 0],
            T_roll=T_roll,
            samp_diff_steps=args.samp_diff_steps,
            ddim_eta=args.ddim_eta,
            c_forward=c_forward,
        )

        error = rollout_error_per_trajectory(
            generated_forward,
            ground_truth_forward,
        )

        forward_errors.append(error)

        print(
            "  rollout MSE:",
            [round(value, 6) for value in error.tolist()],
        )

    forward_errors = torch.stack(
        forward_errors,
        dim=0,
    )

    forward_mean = forward_errors.mean(dim=0)

    forward_std = forward_errors.std(
        dim=0,
        unbiased=False,
    )

    # -------------------------------------------------------
    # Cyclic candidates
    # -------------------------------------------------------
    cyclic_scores = []
    cyclic_true_errors = []

    for candidate_index in range(args.K_cyclic):
        print(
            f"\nCyclic candidate "
            f"{candidate_index + 1}/{args.K_cyclic}"
        )

        forward_trajectory = rollout_forward(
            model=model,
            x0=ground_truth_forward[:, 0],
            T_roll=T_roll,
            samp_diff_steps=args.samp_diff_steps,
            ddim_eta=args.ddim_eta,
            c_forward=c_forward,
        )

        backward_trajectory = rollout_backward(
            model=model,
            xT=forward_trajectory[:, -1],
            T_roll=T_roll,
            samp_diff_steps=args.samp_diff_steps,
            ddim_eta=args.ddim_eta,
            c_backward=c_backward,
        )

        forward_reversed = torch.flip(
            forward_trajectory,
            dims=[1],
        )

        # Per-frame cyclic MSE: (N, T)
        path_cyclic_error = (
            (backward_trajectory - forward_reversed) ** 2
        ).mean(dim=(2, 3, 4))

        # One path-wise cyclic score per trajectory: (N,)
        path_cyclic_score = path_cyclic_error.mean(dim=1)

        # Ground-truth rollout error of this forward candidate: (N,)
        true_rollout_error = rollout_error_per_trajectory(
            forward_trajectory,
            ground_truth_forward,
        )

        cyclic_scores.append(path_cyclic_score)
        cyclic_true_errors.append(true_rollout_error)

        print(
            "  path cyclic score:",
            [
                round(value, 6)
                for value in path_cyclic_score.tolist()
            ],
        )

        print(
            "  ground-truth rollout MSE:",
            [
                round(value, 6)
                for value in true_rollout_error.tolist()
            ],
        )

    cyclic_scores = torch.stack(
        cyclic_scores,
        dim=0,
    )

    cyclic_true_errors = torch.stack(
        cyclic_true_errors,
        dim=0,
    )

    # Select the minimum cyclic-score candidate independently
    # for each of the selected ground-truth trajectories.
    best_cyclic_index = torch.argmin(
        cyclic_scores,
        dim=0,
    )

    best_cyclic_error = torch.zeros(
        len(trajectory_indices),
        dtype=torch.float32,
    )

    for trajectory_position in range(len(trajectory_indices)):
        selected_candidate = int(
            best_cyclic_index[trajectory_position].item()
        )

        best_cyclic_error[trajectory_position] = (
            cyclic_true_errors[
                selected_candidate,
                trajectory_position,
            ]
        )

    # -------------------------------------------------------
    # Averages over the selected trajectories
    # -------------------------------------------------------
    dataset_forward_mean = forward_mean.mean()

    dataset_forward_std = forward_mean.std(
        unbiased=False,
    )

    dataset_cyclic_mean = best_cyclic_error.mean()

    dataset_cyclic_std = best_cyclic_error.std(
        unbiased=False,
    )

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------
    tag = (
        f"dit_{model_size}_{dataset_name}"
        if dit_arch == "legacy"
        else f"dit_{dit_arch}_{model_size}_{dataset_name}"
    )

    suffix = (
        "all_trajs"
        if args.all_trajs
        else "selected_trajs"
    )

    base_name = (
        f"overleaf_table_{tag}_"
        f"forward{args.K_forward}_"
        f"cyclic{args.K_cyclic}_"
        f"steps{args.samp_diff_steps}_"
        f"{suffix}"
    )

    pt_path = os.path.join(
        args.out_dir,
        f"{base_name}.pt",
    )

    txt_path = os.path.join(
        args.out_dir,
        f"{base_name}.txt",
    )

    csv_path = os.path.join(
        args.out_dir,
        f"{base_name}.csv",
    )

    torch.save(
        {
            "trajectory_indices": trajectory_indices,
            "forward_errors": forward_errors,
            "forward_mean": forward_mean,
            "forward_std": forward_std,
            "cyclic_scores": cyclic_scores,
            "cyclic_true_errors": cyclic_true_errors,
            "best_cyclic_index": best_cyclic_index,
            "best_cyclic_error": best_cyclic_error,
            "dataset_forward_mean": dataset_forward_mean,
            "dataset_forward_std": dataset_forward_std,
            "dataset_cyclic_mean": dataset_cyclic_mean,
            "dataset_cyclic_std": dataset_cyclic_std,
            "K_forward": args.K_forward,
            "K_cyclic": args.K_cyclic,
            "T_roll": T_roll,
            "samp_diff_steps": args.samp_diff_steps,
            "ddim_eta": args.ddim_eta,
            "metric": "trajectory_mse_mean_over_T_C_H_W",
            "checkpoint": checkpoint_path,
            "dataset": dataset_path,
        },
        pt_path,
    )

    with open(txt_path, "w") as file:
        file.write(f"Model: {tag}\n")
        file.write(
            "Metric: trajectory MSE averaged over T,C,H,W\n"
        )
        file.write(f"K_forward: {args.K_forward}\n")
        file.write(f"K_cyclic: {args.K_cyclic}\n")
        file.write(
            f"samp_diff_steps: {args.samp_diff_steps}\n"
        )
        file.write(f"ddim_eta: {args.ddim_eta}\n")
        file.write(
            f"Trajectory indices: {trajectory_indices}\n\n"
        )

        file.write(
            f"Forward row: E_for,K={args.K_forward} "
            "= mean ± std\n"
        )

        for position, trajectory_index in enumerate(
            trajectory_indices
        ):
            file.write(
                f"tau_n={trajectory_index}: "
                f"{forward_mean[position].item():.6f} ± "
                f"{forward_std[position].item():.6f}\n"
            )

        file.write(
            f"\nCyclic row: "
            f"E_best_cyclic,K={args.K_cyclic}\n"
        )

        for position, trajectory_index in enumerate(
            trajectory_indices
        ):
            file.write(
                f"tau_n={trajectory_index}: "
                f"{best_cyclic_error[position].item():.6f} "
                f"(best candidate "
                f"{best_cyclic_index[position].item()})\n"
            )

        file.write("\nDataset-level averages:\n")

        file.write(
            f"Forward: "
            f"{dataset_forward_mean.item():.6f} ± "
            f"{dataset_forward_std.item():.6f}\n"
        )

        file.write(
            f"Cyclic: "
            f"{dataset_cyclic_mean.item():.6f} ± "
            f"{dataset_cyclic_std.item():.6f}\n"
        )

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "trajectory_index",
                "forward_mean",
                "forward_std",
                "best_cyclic_error",
                "best_cyclic_candidate",
                "K_forward",
                "K_cyclic",
                "samp_diff_steps",
                "ddim_eta",
            ]
        )

        for position, trajectory_index in enumerate(
            trajectory_indices
        ):
            writer.writerow(
                [
                    tag,
                    trajectory_index,
                    forward_mean[position].item(),
                    forward_std[position].item(),
                    best_cyclic_error[position].item(),
                    best_cyclic_index[position].item(),
                    args.K_forward,
                    args.K_cyclic,
                    args.samp_diff_steps,
                    args.ddim_eta,
                ]
            )

        writer.writerow([])

        writer.writerow(
            [
                "dataset_level",
                "all_selected_trajectories",
                dataset_forward_mean.item(),
                dataset_forward_std.item(),
                dataset_cyclic_mean.item(),
                dataset_cyclic_std.item(),
                args.K_forward,
                args.K_cyclic,
                args.samp_diff_steps,
                args.ddim_eta,
            ]
        )

    print("\nSaved:")
    print(pt_path)
    print(txt_path)
    print(csv_path)

    print("\n===== DIT TABLE VALUES =====")

    print(
        f"\nForward row: E_for,K={args.K_forward}"
    )

    for position, trajectory_index in enumerate(
        trajectory_indices
    ):
        print(
            f"tau_n={trajectory_index}: "
            f"{forward_mean[position].item():.6f} ± "
            f"{forward_std[position].item():.6f}"
        )

    print(
        f"\nCyclic row: "
        f"E_best_cyclic,K={args.K_cyclic}"
    )

    for position, trajectory_index in enumerate(
        trajectory_indices
    ):
        print(
            f"tau_n={trajectory_index}: "
            f"{best_cyclic_error[position].item():.6f} "
            f"(best candidate "
            f"{best_cyclic_index[position].item()})"
        )

    print("\nDataset-level averages:")

    print(
        f"Forward: "
        f"{dataset_forward_mean.item():.6f} ± "
        f"{dataset_forward_std.item():.6f}"
    )

    print(
        f"Cyclic: "
        f"{dataset_cyclic_mean.item():.6f} ± "
        f"{dataset_cyclic_std.item():.6f}"
    )


if __name__ == "__main__":
    main()