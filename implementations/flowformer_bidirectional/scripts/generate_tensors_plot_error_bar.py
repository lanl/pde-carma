import os
import sys
import csv
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


def resolve_path(path):
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


def rollout_forward(
    model,
    x0,
    T_roll,
    num_ode_steps,
    c_forward,
):
    """
    Generate an autoregressive forward rollout.

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
        path = model.sample(
            x_cond=condition,
            num_ode_steps=num_ode_steps,
            c=c_forward,
        )

        x_next = path[:, -1].detach().cpu()

        trajectory[:, t] = x_next
        condition = x_next

    return trajectory


def rollout_backward(
    model,
    xT,
    T_roll,
    num_ode_steps,
    c_backward,
):
    """
    Generate an autoregressive backward rollout.

    xT:
        (N, C, H, W)

    returns:
        (N, T_roll, C, H, W)

    The returned order is:
        terminal frame,
        previous frame,
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
        path = model.sample(
            x_cond=condition,
            num_ode_steps=num_ode_steps,
            c=c_backward,
        )

        x_previous = path[:, -1].detach().cpu()

        trajectory[:, t] = x_previous
        condition = x_previous

    return trajectory


def rollout_mse(predicted, ground_truth):
    """
    Compute rollout MSE per trajectory.

    The observed initial frame t=0 is excluded.
    """
    return (
        (predicted[:, 1:] - ground_truth[:, 1:]) ** 2
    ).mean(dim=(1, 2, 3, 4))


def summarize_across_dataset(values):
    """
    values:
        (number_of_K_values, number_of_test_trajectories)

    Returns:
        mean, std, lower, upper
        each with shape (number_of_K_values,)
    """
    mean = values.mean(dim=1)

    std = values.std(
        dim=1,
        unbiased=False,
    )

    lower = mean - std
    upper = mean + std

    return mean, std, lower, upper


def plot_dataset_metric(
    K_values,
    values,
    ylabel,
    title,
    save_path,
):
    """
    Plot dataset mean ± one standard deviation.

    values:
        torch.Tensor or numpy array with shape
        (number_of_K_values, number_of_test_trajectories)
    """
    if torch.is_tensor(values):
        values = values.detach().cpu()

        mean, std, lower, upper = summarize_across_dataset(
            values
        )

        mean = mean.numpy()
        std = std.numpy()
        lower = lower.numpy()
        upper = upper.numpy()

    else:
        values = np.asarray(values)

        mean = values.mean(axis=1)
        std = values.std(axis=1)
        lower = mean - std
        upper = mean + std

    K_array = np.asarray(K_values)

    plt.figure(figsize=(7.2, 4.8))

    # Dataset-level mean.
    plt.plot(
        K_array,
        mean,
        marker="o",
        linewidth=2.5,
        label="Dataset mean",
    )

    # Upper and lower variability boundaries.
    plt.plot(
        K_array,
        upper,
        linestyle="--",
        linewidth=1.4,
        label=r"Mean $+\sigma$",
    )

    plt.plot(
        K_array,
        lower,
        linestyle="--",
        linewidth=1.4,
        label=r"Mean $-\sigma$",
    )

    # Shaded mean ± standard-deviation region.
    plt.fill_between(
        K_array,
        lower,
        upper,
        alpha=0.20,
        label=r"Mean $\pm\sigma$",
    )

    plt.xlabel("Number of cyclic candidates, K")
    plt.ylabel(ylabel)
    plt.xticks(K_values)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot dataset-level cyclic metrics versus the number "
            "of candidate runs."
        )
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/vids_test_ce-rm.npy",
    )

    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_T_rollout",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--K_values",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
    )

    parser.add_argument(
        "--device_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--traj_batch_size",
        type=int,
        default=4,
        help=(
            "Number of test trajectories processed together. "
            "Reduce this if memory use is too high."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/cyclic_vs_runs_dataset",
    )

    args = parser.parse_args()

    if args.num_ode_steps <= 0:
        raise ValueError("--num_ode_steps must be positive.")

    if args.traj_batch_size <= 0:
        raise ValueError("--traj_batch_size must be positive.")

    if any(K <= 0 for K in args.K_values):
        raise ValueError("Every entry in --K_values must be positive.")

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    checkpoint_path = resolve_path(args.ckpt)
    dataset_path = resolve_path(args.dataset)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    config = checkpoint["args"]

    if isinstance(config, dict):
        model_size = config["ms"]
        dataset_name = config["ds"]
    else:
        model_size = config.ms
        dataset_name = config.ds

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    videos_test = torch.from_numpy(
        np.load(
            dataset_path,
            mmap_mode="r",
        )
    ).float()

    if videos_test.ndim != 5:
        raise ValueError(
            "Expected test data with shape (N,T,C,H,W), "
            f"but received {tuple(videos_test.shape)}."
        )

    N_total, T_total, C, H, W = videos_test.shape

    T_roll = min(
        args.max_T_rollout,
        T_total,
    )

    K_values = sorted(
        set(args.K_values)
    )

    K_max = max(K_values)

    c_forward = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float32,
    )

    c_backward = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float32,
    )

    tag = f"flowformer_{model_size}_{dataset_name}"

    print("===================================")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {tag}")
    print(f"Full test-set shape: {tuple(videos_test.shape)}")
    print(f"Number of test trajectories: {N_total}")
    print(f"T_roll: {T_roll}")
    print(f"ODE steps: {args.num_ode_steps}")
    print(f"K values: {K_values}")
    print(f"Maximum candidate pool: {K_max}")
    print(f"Trajectory batch size: {args.traj_batch_size}")
    print(f"Device: {device}")
    print("===================================")

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=checkpoint_path,
        img_size=(H, W),
        device=device,
    )

    # Scores for every candidate and every test trajectory.
    #
    # Shape:
    #     (K_max, N_total)
    endpoint_scores = torch.empty(
        K_max,
        N_total,
        dtype=torch.float32,
    )

    path_scores = torch.empty(
        K_max,
        N_total,
        dtype=torch.float32,
    )

    true_errors = torch.empty(
        K_max,
        N_total,
        dtype=torch.float32,
    )

    # -------------------------------------------------------
    # Generate K_max candidates for the entire test dataset.
    #
    # Test trajectories are processed in small batches to
    # avoid holding the full dataset rollout in memory.
    # -------------------------------------------------------
    for candidate_index in range(K_max):
        print(
            f"\n===== Candidate "
            f"{candidate_index + 1}/{K_max} ====="
        )

        for batch_start in range(
            0,
            N_total,
            args.traj_batch_size,
        ):
            batch_end = min(
                batch_start + args.traj_batch_size,
                N_total,
            )

            print(
                f"Processing trajectories "
                f"{batch_start}:{batch_end}"
            )

            real_forward = (
                videos_test[
                    batch_start:batch_end,
                    :T_roll,
                ]
                .detach()
                .cpu()
            )

            forward_trajectory = rollout_forward(
                model=model,
                x0=real_forward[:, 0],
                T_roll=T_roll,
                num_ode_steps=args.num_ode_steps,
                c_forward=c_forward,
            )

            backward_trajectory = rollout_backward(
                model=model,
                xT=forward_trajectory[:, -1],
                T_roll=T_roll,
                num_ode_steps=args.num_ode_steps,
                c_backward=c_backward,
            )

            forward_reversed = torch.flip(
                forward_trajectory,
                dims=[1],
            )

            endpoint_error = (
                (
                    backward_trajectory[:, -1]
                    - forward_reversed[:, -1]
                )
                ** 2
            ).mean(dim=(1, 2, 3))

            path_error = (
                (
                    backward_trajectory
                    - forward_reversed
                )
                ** 2
            ).mean(dim=(1, 2, 3, 4))

            true_error = rollout_mse(
                forward_trajectory,
                real_forward,
            )

            endpoint_scores[
                candidate_index,
                batch_start:batch_end,
            ] = endpoint_error

            path_scores[
                candidate_index,
                batch_start:batch_end,
            ] = path_error

            true_errors[
                candidate_index,
                batch_start:batch_end,
            ] = true_error

            del real_forward
            del forward_trajectory
            del backward_trajectory
            del forward_reversed
            del endpoint_error
            del path_error
            del true_error

    # -------------------------------------------------------
    # Compute best candidate among the first K samples for
    # every test trajectory independently.
    # -------------------------------------------------------
    best_path_by_K = []
    best_endpoint_by_K = []
    selected_true_error_by_K = []
    best_candidate_index_by_K = []

    trajectory_positions = torch.arange(N_total)

    for K in K_values:
        path_subset = path_scores[:K]
        endpoint_subset = endpoint_scores[:K]
        true_subset = true_errors[:K]

        best_candidate_index = torch.argmin(
            path_subset,
            dim=0,
        )

        best_path = path_subset[
            best_candidate_index,
            trajectory_positions,
        ]

        best_endpoint = endpoint_subset[
            best_candidate_index,
            trajectory_positions,
        ]

        selected_true_error = true_subset[
            best_candidate_index,
            trajectory_positions,
        ]

        best_path_by_K.append(best_path)
        best_endpoint_by_K.append(best_endpoint)
        selected_true_error_by_K.append(
            selected_true_error
        )
        best_candidate_index_by_K.append(
            best_candidate_index
        )

    best_path_by_K = torch.stack(
        best_path_by_K,
        dim=0,
    )

    best_endpoint_by_K = torch.stack(
        best_endpoint_by_K,
        dim=0,
    )

    selected_true_error_by_K = torch.stack(
        selected_true_error_by_K,
        dim=0,
    )

    best_candidate_index_by_K = torch.stack(
        best_candidate_index_by_K,
        dim=0,
    )

    # -------------------------------------------------------
    # Random and Oracle errors from the SAME candidate pool
    # -------------------------------------------------------

    random_true_error_by_K = torch.stack(
        [true_errors[:K].mean(dim=0) for K in K_values],
        dim=0,
    )

    oracle_true_error_by_K = torch.stack(
        [true_errors[:K].min(dim=0).values for K in K_values],
        dim=0,
    )

    # -------------------------------------------------------
    # Dataset-level statistics
    # -------------------------------------------------------
    path_mean, path_std, path_lower, path_upper = (
        summarize_across_dataset(best_path_by_K)
    )

    endpoint_mean, endpoint_std, endpoint_lower, endpoint_upper = (
        summarize_across_dataset(best_endpoint_by_K)
    )

    rollout_mean, rollout_std, rollout_lower, rollout_upper = (
        summarize_across_dataset(
            selected_true_error_by_K
        )
    )
    random_mean, random_std, _, _ = (
        summarize_across_dataset(
            random_true_error_by_K
        )
    )

    oracle_mean, oracle_std, _, _ = (
        summarize_across_dataset(
            oracle_true_error_by_K
        )
    )

    delta_cyc = (
        100.0
        * (random_mean - rollout_mean)
        / random_mean
    )

    # -------------------------------------------------------
    # Save tensor results
    # -------------------------------------------------------
    base_name = (
        f"dataset_cyclic_vs_runs_{tag}_"
        f"ode{args.num_ode_steps}"
    )

    pt_path = os.path.join(
        args.out_dir,
        f"{base_name}.pt",
    )

    csv_path = os.path.join(
        args.out_dir,
        f"{base_name}.csv",
    )

    torch.save(
        {
            "K_values": K_values,
            "num_test_trajectories": N_total,
            "num_ode_steps": args.num_ode_steps,
            "T_roll": T_roll,
            "endpoint_scores_all_candidates": endpoint_scores,
            "path_scores_all_candidates": path_scores,
            "true_errors_all_candidates": true_errors,
            "best_candidate_index_by_K": (
                best_candidate_index_by_K
            ),
            "best_path_by_K": best_path_by_K,
            "best_endpoint_by_K": best_endpoint_by_K,
            "selected_true_error_by_K": (
                selected_true_error_by_K
            ),
            "path_mean": path_mean,
            "path_std": path_std,
            "path_lower": path_lower,
            "path_upper": path_upper,
            "endpoint_mean": endpoint_mean,
            "endpoint_std": endpoint_std,
            "endpoint_lower": endpoint_lower,
            "endpoint_upper": endpoint_upper,
            "rollout_mean": rollout_mean,
            "rollout_std": rollout_std,
            "rollout_lower": rollout_lower,
            "rollout_upper": rollout_upper,
        },
        pt_path,
    )

    print(f"\nSaved tensor results: {pt_path}")

    # -------------------------------------------------------
    # Save dataset statistics as CSV
    # -------------------------------------------------------
    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "dataset",
                "ode_steps",
                "K",
                "num_test_trajectories",
                "path_mean",
                "path_std",
                "path_lower",
                "path_upper",
                "endpoint_mean",
                "endpoint_std",
                "endpoint_lower",
                "endpoint_upper",
                "selected_rollout_mean",
                "selected_rollout_std",
                "selected_rollout_lower",
                "selected_rollout_upper",
            ]
        )

        for row_index, K in enumerate(K_values):
            writer.writerow(
                [
                    tag,
                    dataset_name,
                    args.num_ode_steps,
                    K,
                    N_total,
                    path_mean[row_index].item(),
                    path_std[row_index].item(),
                    path_lower[row_index].item(),
                    path_upper[row_index].item(),
                    endpoint_mean[row_index].item(),
                    endpoint_std[row_index].item(),
                    endpoint_lower[row_index].item(),
                    endpoint_upper[row_index].item(),
                    rollout_mean[row_index].item(),
                    rollout_std[row_index].item(),
                    rollout_lower[row_index].item(),
                    rollout_upper[row_index].item(),
                ]
            )

    print(f"Saved CSV results: {csv_path}")

    # -------------------------------------------------------
    # Generate dataset-level plots
    # -------------------------------------------------------
    plot_dataset_metric(
        K_values=K_values,
        values=best_path_by_K,
        ylabel=r"Best path-wise cyclic loss",
        title=(
            f"Path-wise cyclic loss vs candidates\n"
            f"{tag}, all {N_total} test trajectories"
        ),
        save_path=os.path.join(
            args.out_dir,
            f"path_cyclic_mean_std_vs_runs_"
            f"{tag}_ode{args.num_ode_steps}.png",
        ),
    )

    plot_dataset_metric(
        K_values=K_values,
        values=best_endpoint_by_K,
        ylabel=r"Endpoint cyclic loss",
        title=(
            f"Endpoint cyclic loss vs candidates\n"
            f"{tag}, all {N_total} test trajectories"
        ),
        save_path=os.path.join(
            args.out_dir,
            f"endpoint_cyclic_mean_std_vs_runs_"
            f"{tag}_ode{args.num_ode_steps}.png",
        ),
    )

    plot_dataset_metric(
        K_values=K_values,
        values=selected_true_error_by_K,
        ylabel="Selected rollout MSE",
        title=(
            f"Selected rollout error vs candidates\n"
            f"{tag}, all {N_total} test trajectories"
        ),
        save_path=os.path.join(
            args.out_dir,
            f"selected_rollout_mean_std_vs_runs_"
            f"{tag}_ode{args.num_ode_steps}.png",
        ),
    )

    # -------------------------------------------------------
    # Print dataset-level values
    # -------------------------------------------------------
    print("\n===== DATASET-LEVEL VALUES =====")
    print(f"Number of test trajectories: {N_total}")

    for row_index, K in enumerate(K_values):
        print(f"\nK={K}")

        print(
            "  Path-wise cyclic loss: "
            f"{path_mean[row_index].item():.6f} ± "
            f"{path_std[row_index].item():.6f}"
        )

        print(
            "  Endpoint cyclic loss: "
            f"{endpoint_mean[row_index].item():.6f} ± "
            f"{endpoint_std[row_index].item():.6f}"
        )

        print(
            "  Selected rollout MSE: "
            f"{rollout_mean[row_index].item():.6f} ± "
            f"{rollout_std[row_index].item():.6f}"
        )

    print("\n===== RANDOM / CYCLIC / ORACLE =====")

    for row_index, K in enumerate(K_values):
        print(f"\nK = {K}")

        print(
            f"E_random = "
            f"{random_mean[row_index].item():.6f} ± "
            f"{random_std[row_index].item():.6f}"
        )

        print(
            f"E_cyclic = "
            f"{rollout_mean[row_index].item():.6f} ± "
            f"{rollout_std[row_index].item():.6f}"
        )

        print(
            f"E_oracle = "
            f"{oracle_mean[row_index].item():.6f} ± "
            f"{oracle_std[row_index].item():.6f}"
        )

        print(
            f"Delta_cyc = "
            f"{delta_cyc[row_index].item():.2f}%"
        )


if __name__ == "__main__":
    main()