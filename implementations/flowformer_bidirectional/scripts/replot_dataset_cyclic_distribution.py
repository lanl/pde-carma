import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    results = torch.load(path, map_location="cpu")

    required_keys = [
        "K_values",
        "best_path_by_K",
    ]

    missing = [
        key for key in required_keys
        if key not in results
    ]

    if missing:
        raise KeyError(
            "Results file is missing required fields: "
            + ", ".join(missing)
        )

    return results


def infer_tag(results, input_path):
    if "tag" in results:
        return str(results["tag"])

    filename = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    prefix = "dataset_cyclic_vs_runs_"

    if filename.startswith(prefix):
        filename = filename[len(prefix):]

    return filename


def compute_statistics(values):
    """
    Parameters
    ----------
    values:
        Array with shape:
        (number_of_K_values, number_of_test_trajectories)

    Returns
    -------
    Dictionary containing dataset statistics for every K.
    """
    values = np.asarray(values, dtype=np.float64)

    return {
        "mean": values.mean(axis=1),
        "median": np.median(values, axis=1),
        "minimum": values.min(axis=1),
        "maximum": values.max(axis=1),
        "q25": np.percentile(values, 25, axis=1),
        "q75": np.percentile(values, 75, axis=1),
        "std": values.std(axis=1, ddof=1),
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
):
    """
    For every K:
      1. plot a vertical line from minimum to maximum,
      2. plot every test-trajectory error along that line,
      3. mark the mean,
      4. mark the median.

    values shape:
        (number_of_K_values, number_of_test_trajectories)
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(
            "values must have shape "
            "(number_of_K_values, number_of_test_trajectories), "
            f"but received {values.shape}"
        )

    if values.shape[0] != len(K_values):
        raise ValueError(
            "The number of rows in values must match "
            "the number of K values."
        )

    statistics = compute_statistics(values)
    rng = np.random.default_rng(random_seed)

    K_array = np.asarray(K_values, dtype=np.float64)

    plt.figure(figsize=(8.0, 5.2))

    for row_index, K in enumerate(K_array):
        trajectory_errors = values[row_index]

        minimum = statistics["minimum"][row_index]
        maximum = statistics["maximum"][row_index]
        # mean = statistics["mean"][row_index]
        # median = statistics["median"][row_index]

        # Full observed range across all test trajectories.
        plt.vlines(
            x=K,
            ymin=minimum,
            ymax=maximum,
            linewidth=2.2,
            alpha=0.80,
            zorder=1,
        )

        # Slight horizontal jitter makes overlapping values visible.
        x_jitter = rng.uniform(
            low=-jitter_width,
            high=jitter_width,
            size=trajectory_errors.shape[0],
        )

        plt.scatter(
            np.full_like(trajectory_errors, K) + x_jitter,
            trajectory_errors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )

        # # Dataset mean.
        # plt.scatter(
        #     K,
        #     mean,
        #     marker="D",
        #     s=68,
        #     edgecolors="black",
        #     linewidths=0.8,
        #     label="Dataset mean" if row_index == 0 else None,
        #     zorder=4,
        # )

        # # Dataset median shown as a short horizontal segment.
        # plt.hlines(
        #     y=median,
        #     xmin=K - 0.20,
        #     xmax=K + 0.20,
        #     linewidth=2.4,
        #     label="Dataset median" if row_index == 0 else None,
        #     zorder=3,
        # )

    # Connect means to show the dataset-level trend.
    plt.plot(
        K_array,
        statistics["mean"],
        linewidth=2.0,
        marker="D",
        label="Mean trend",
        zorder=3,
    )

    # Maximum trajectory error.
    plt.plot(
        K_array,
        statistics["maximum"],
        linewidth=1.8,
        linestyle="--",
        marker="^",
        label="Max",
        zorder=3,
    )

    # Minimum trajectory error.
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
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot: {save_path}")

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the full distribution of best path-wise cyclic "
            "errors across all test trajectories at each K."
        )
    )

    parser.add_argument(
        "--input_pt",
        type=str,
        required=True,
        help=(
            "Previously saved dataset-level .pt file from "
            "plot_dataset_cyclic_metrics_vs_runs.py"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/cyclic_vs_runs_distribution",
    )

    parser.add_argument(
        "--title_prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--jitter_width",
        type=float,
        default=0.12,
        help=(
            "Horizontal jitter around each K. This changes only "
            "visual placement, not the metric values."
        ),
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=13,
    )

    parser.add_argument(
        "--point_alpha",
        type=float,
        default=0.42,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed used only for visual jitter.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = load_results(args.input_pt)

    K_values = list(results["K_values"])

    best_path_by_K = results["best_path_by_K"]

    if torch.is_tensor(best_path_by_K):
        best_path_by_K = (
            best_path_by_K
            .detach()
            .cpu()
            .numpy()
        )
    else:
        best_path_by_K = np.asarray(best_path_by_K)

    if best_path_by_K.ndim != 2:
        raise ValueError(
            "best_path_by_K must have shape "
            "(number_of_K_values, number_of_test_trajectories), "
            f"but received {best_path_by_K.shape}"
        )

    number_of_trajectories = best_path_by_K.shape[1]

    tag = infer_tag(
        results=results,
        input_path=args.input_pt,
    )

    title_prefix = (
        args.title_prefix
        if args.title_prefix is not None
        else tag
    )

    safe_tag = (
        tag.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    print("===================================")
    print(f"Input file: {args.input_pt}")
    print(f"Experiment: {tag}")
    print(f"K values: {K_values}")
    print(
        f"Number of test trajectories: "
        f"{number_of_trajectories}"
    )
    print("===================================")

    save_path = os.path.join(
        args.out_dir,
        f"path_cyclic_distribution_vs_runs_{safe_tag}.png",
    )

    statistics = plot_vertical_distributions(
        K_values=K_values,
        values=best_path_by_K,
        ylabel="Best path-wise cyclic loss",
        title=(
            "Distribution of best path-wise cyclic loss\n"
            f"{title_prefix}, all "
            f"{number_of_trajectories} test trajectories"
        ),
        save_path=save_path,
        jitter_width=args.jitter_width,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        random_seed=args.random_seed,
    )

    csv_path = os.path.join(
        args.out_dir,
        f"path_cyclic_distribution_stats_{safe_tag}.csv",
    )

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
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

    print("\n===== DISTRIBUTION SUMMARY =====")

    for row_index, K in enumerate(K_values):
        minimum = statistics["minimum"][row_index]
        maximum = statistics["maximum"][row_index]

        print(f"\nK={K}")
        print(
            f"  Mean:   "
            f"{statistics['mean'][row_index]:.6f}"
        )
        print(
            f"  Median: "
            f"{statistics['median'][row_index]:.6f}"
        )
        print(
            f"  Min:    {minimum:.6f}"
        )
        print(
            f"  Max:    {maximum:.6f}"
        )
        print(
            f"  Range:  "
            f"{maximum - minimum:.6f}"
        )
        print(
            f"  IQR:    "
            f"{statistics['q25'][row_index]:.6f} to "
            f"{statistics['q75'][row_index]:.6f}"
        )


if __name__ == "__main__":
    main()