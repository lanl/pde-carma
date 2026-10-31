#!/usr/bin/env python3

"""
Replot the full-test-set distribution of the best DiT path-wise cyclic
error for each candidate-pool size K.

This script reads the .pt file created by:

    generate_tensors_plot_error_bar_dit.py

It does not run inference again. It only loads the saved tensors, produces a
publication-quality distribution plot, and exports descriptive statistics.
"""

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    # weights_only=False is needed for general experiment dictionaries on
    # newer PyTorch versions. Older versions do not accept this argument.
    try:
        results = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        results = torch.load(path, map_location="cpu")

    if not isinstance(results, dict):
        raise TypeError(
            "Expected the input .pt file to contain a dictionary, but "
            f"received {type(results).__name__}."
        )

    required_keys = [
        "K_values",
        "best_path_by_K",
    ]

    missing = [key for key in required_keys if key not in results]

    if missing:
        raise KeyError(
            "Results file is missing required fields: "
            + ", ".join(missing)
            + "\nUse the .pt file produced by "
            "generate_tensors_plot_error_bar_dit.py."
        )

    return results


def sanitize_filename(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_") or "dit_experiment"


def infer_tag(results, input_path):
    """Infer a readable DiT experiment tag from metadata or filename."""
    if "tag" in results:
        return str(results["tag"])

    filename = os.path.splitext(os.path.basename(input_path))[0]
    prefix = "dataset_cyclic_vs_runs_"

    if filename.startswith(prefix):
        filename = filename[len(prefix):]

    return filename


def infer_title_metadata(results, tag):
    diffusion_steps = results.get("samp_diff_steps")
    ddim_eta = results.get("ddim_eta")
    num_trajectories = results.get("num_test_trajectories")

    pieces = [tag]

    if diffusion_steps is not None:
        pieces.append(f"sampling steps={diffusion_steps}")

    if ddim_eta is not None:
        pieces.append(f"DDIM eta={float(ddim_eta):g}")

    if num_trajectories is not None:
        pieces.append(f"N={int(num_trajectories)}")

    return ", ".join(pieces)


def to_numpy_2d(value, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    value = np.asarray(value, dtype=np.float64)

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape "
            "(number_of_K_values, number_of_test_trajectories), "
            f"but received {value.shape}."
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return value


def compute_statistics(values):
    """Compute descriptive statistics across test trajectories for each K."""
    values = np.asarray(values, dtype=np.float64)
    number_of_trajectories = values.shape[1]

    # Use the sample standard deviation when possible. For a single
    # trajectory, return zero instead of NaN.
    ddof = 1 if number_of_trajectories > 1 else 0

    return {
        "mean": values.mean(axis=1),
        "median": np.median(values, axis=1),
        "minimum": values.min(axis=1),
        "maximum": values.max(axis=1),
        "q25": np.percentile(values, 25, axis=1),
        "q75": np.percentile(values, 75, axis=1),
        "std": values.std(axis=1, ddof=ddof),
    }


def plot_vertical_distributions(
    K_values,
    values,
    title,
    ylabel,
    save_path,
    jitter_width=0.12,
    point_size=13,
    point_alpha=0.42,
    random_seed=0,
    log_y=False,
):
    """
    For each K, plot every test-trajectory value with horizontal jitter,
    the observed min-to-max range, and connected mean/min/max trends.
    """
    values = to_numpy_2d(values, "values")

    if values.shape[0] != len(K_values):
        raise ValueError(
            "The number of rows in values must match the number of K values. "
            f"Received {values.shape[0]} rows and {len(K_values)} K values."
        )

    if jitter_width < 0:
        raise ValueError("--jitter_width must be non-negative.")

    if point_size <= 0:
        raise ValueError("--point_size must be positive.")

    if not 0 <= point_alpha <= 1:
        raise ValueError("--point_alpha must be between 0 and 1.")

    statistics = compute_statistics(values)
    rng = np.random.default_rng(random_seed)
    K_array = np.asarray(K_values, dtype=np.float64)

    plt.figure(figsize=(8.0, 5.2))

    for row_index, K in enumerate(K_array):
        trajectory_errors = values[row_index]
        minimum = statistics["minimum"][row_index]
        maximum = statistics["maximum"][row_index]

        plt.vlines(
            x=K,
            ymin=minimum,
            ymax=maximum,
            linewidth=2.2,
            alpha=0.80,
            zorder=1,
        )

        x_jitter = rng.uniform(
            low=-jitter_width,
            high=jitter_width,
            size=trajectory_errors.shape[0],
        )

        plt.scatter(
            np.full(trajectory_errors.shape, K, dtype=np.float64) + x_jitter,
            trajectory_errors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )

    plt.plot(
        K_array,
        statistics["mean"],
        linewidth=2.0,
        marker="D",
        label="Mean trend",
        zorder=3,
    )

    plt.plot(
        K_array,
        statistics["maximum"],
        linewidth=1.8,
        linestyle="--",
        marker="^",
        label="Max",
        zorder=3,
    )

    plt.plot(
        K_array,
        statistics["minimum"],
        linewidth=1.8,
        linestyle=":",
        marker="v",
        label="Min",
        zorder=3,
    )

    plt.xlabel("Number of cyclic candidates, K")
    plt.ylabel(ylabel)
    plt.xticks(K_values)
    plt.title(title)

    if log_y:
        if np.any(values <= 0):
            raise ValueError(
                "--log_y was requested, but the plotted values include "
                "zero or negative entries."
            )
        plt.yscale("log")

    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {save_path}")
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replot the full distribution of best DiT path-wise cyclic "
            "errors across all test trajectories for every K."
        )
    )

    parser.add_argument(
        "--input_pt",
        type=str,
        required=True,
        help=(
            "Dataset-level .pt file created by "
            "generate_tensors_plot_error_bar_dit.py."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/dit_cyclic_vs_runs_distribution",
    )

    parser.add_argument(
        "--title_prefix",
        type=str,
        default=None,
        help="Optional title text. By default, metadata is inferred from the .pt file.",
    )

    parser.add_argument(
        "--jitter_width",
        type=float,
        default=0.12,
        help=(
            "Horizontal jitter around each K. This affects only visual "
            "placement, not the values."
        ),
    )

    parser.add_argument("--point_size", type=float, default=13)
    parser.add_argument("--point_alpha", type=float, default=0.42)

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed used only for visual jitter.",
    )

    parser.add_argument(
        "--log_y",
        action="store_true",
        help="Use a logarithmic y-axis when all errors are positive.",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = load_results(args.input_pt)
    K_values = [int(value) for value in results["K_values"]]

    if len(K_values) == 0:
        raise ValueError("K_values is empty.")

    best_path_by_K = to_numpy_2d(
        results["best_path_by_K"],
        "best_path_by_K",
    )

    if best_path_by_K.shape[0] != len(K_values):
        raise ValueError(
            "best_path_by_K has an incompatible number of rows: "
            f"{best_path_by_K.shape[0]} rows for {len(K_values)} K values."
        )

    number_of_trajectories = best_path_by_K.shape[1]
    saved_number = results.get("num_test_trajectories")

    if saved_number is not None and int(saved_number) != number_of_trajectories:
        raise ValueError(
            "The saved num_test_trajectories metadata does not match "
            "best_path_by_K: "
            f"metadata={saved_number}, tensor={number_of_trajectories}."
        )

    tag = infer_tag(results=results, input_path=args.input_pt)

    title_prefix = (
        args.title_prefix
        if args.title_prefix is not None
        else infer_title_metadata(results, tag)
    )

    safe_tag = sanitize_filename(tag)

    print("===================================")
    print(f"Input file: {args.input_pt}")
    print(f"Experiment: {tag}")
    print(f"K values: {K_values}")
    print(f"Number of test trajectories: {number_of_trajectories}")

    if "samp_diff_steps" in results:
        print(f"Diffusion sampling steps: {results['samp_diff_steps']}")

    if "ddim_eta" in results:
        print(f"DDIM eta: {results['ddim_eta']}")

    print("===================================")

    save_path = os.path.join(
        args.out_dir,
        f"path_cyclic_distribution_vs_runs_{safe_tag}.png",
    )

    statistics = plot_vertical_distributions(
        K_values=K_values,
        values=best_path_by_K,
        ylabel="Best path-wise cyclic MSE",
        title=(
            "Distribution of best DiT path-wise cyclic error\n"
            f"{title_prefix}, all {number_of_trajectories} test trajectories"
        ),
        save_path=save_path,
        jitter_width=args.jitter_width,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        random_seed=args.random_seed,
        log_y=args.log_y,
    )

    csv_path = os.path.join(
        args.out_dir,
        f"path_cyclic_distribution_stats_{safe_tag}.csv",
    )

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model_tag",
                "samp_diff_steps",
                "ddim_eta",
                "K",
                "num_test_trajectories",
                "mean",
                "median",
                "std",
                "minimum",
                "q25",
                "q75",
                "maximum",
                "vertical_range",
            ]
        )

        for row_index, K in enumerate(K_values):
            minimum = statistics["minimum"][row_index]
            maximum = statistics["maximum"][row_index]

            writer.writerow(
                [
                    tag,
                    results.get("samp_diff_steps", ""),
                    results.get("ddim_eta", ""),
                    K,
                    number_of_trajectories,
                    statistics["mean"][row_index],
                    statistics["median"][row_index],
                    statistics["std"][row_index],
                    minimum,
                    statistics["q25"][row_index],
                    statistics["q75"][row_index],
                    maximum,
                    maximum - minimum,
                ]
            )

    print(f"Saved CSV: {csv_path}")
    print("\n===== DIT DISTRIBUTION SUMMARY =====")

    for row_index, K in enumerate(K_values):
        minimum = statistics["minimum"][row_index]
        maximum = statistics["maximum"][row_index]

        print(f"\nK={K}")
        print(f"  Mean:   {statistics['mean'][row_index]:.6f}")
        print(f"  Median: {statistics['median'][row_index]:.6f}")
        print(f"  Std:    {statistics['std'][row_index]:.6f}")
        print(f"  Min:    {minimum:.6f}")
        print(f"  Max:    {maximum:.6f}")
        print(f"  Range:  {maximum - minimum:.6f}")
        print(
            "  IQR:    "
            f"{statistics['q25'][row_index]:.6f} to "
            f"{statistics['q75'][row_index]:.6f}"
        )


if __name__ == "__main__":
    main()
