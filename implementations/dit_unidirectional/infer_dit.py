import csv
import os
import sys
import time

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.append(project_root)

from utils.my_model import DiTInference
from utils.infer_args import InferArgsConfig
from utils.device_manager import DeviceManager
from utils.get_results import TestRolloutVisualizer


def resolve_checkpoint_path(ckpt):
    if os.path.isfile(ckpt):
        return ckpt

    ckpt_path = os.path.join(project_root, "models", ckpt)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


def synchronize_device(device):
    """Synchronize accelerator operations for accurate timing."""
    device_type = torch.device(device).type

    if device_type == "cuda":
        torch.cuda.synchronize()

    elif device_type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def append_to_csv(csv_path, row):
    """Append one experiment result to the table CSV."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = InferArgsConfig()

    # Backward-compatible fallback.
    cyclic_runs = getattr(args, "cyclic_runs", [1])

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    print(f"Using device: {device}")

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    dataset_root = (
        args.dataset_root
        if args.dataset_root
        else project_root
    )

    dataset_dir = os.path.join(dataset_root, "datasets")

    ckpt_path = resolve_checkpoint_path(args.ckpt)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint["args"]

    # Support checkpoints that store args as either dict or Namespace.
    if isinstance(config, dict):
        ds = config["ds"]
        ms = config["ms"]
    else:
        ds = config.ds
        ms = config.ms

    test_path = os.path.join(
        dataset_dir,
        f"vids_test_{ds}.npy",
    )

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Test data not found: {test_path}"
        )

    vids_test = torch.from_numpy(
        np.load(test_path)
    ).float()

    N, T, C, H, W = vids_test.shape
    img_size = (H, W)

    print(f"Test videos shape: {vids_test.shape}")
    print(f"Image size: {img_size}")
    print(f"Model size: {ms}")
    print(f"Dataset: {ds}")
    print(f"Diffusion steps: {args.samp_diff_steps}")
    print(f"DDIM eta: {args.ddim_eta}")
    print(f"Cyclic candidate runs: {cyclic_runs}")

    model = DiTInference.from_pretrained(
        checkpoint_path=ckpt_path,
        img_size=img_size,
        device=device,
    )

    table_csv_path = os.path.join(
        results_dir,
        "dit_diffusion_step_table.csv",
    )

    for run in range(args.num_runs):
        run_number = run + 1

        print(
            f"\n=== Run {run_number}/{args.num_runs} ==="
        )

        visualizer = TestRolloutVisualizer(
            model=model,
            vids_test=vids_test,

            # For DiT, this controls diffusion sampling steps.
            num_ode_steps=args.samp_diff_steps,

            ddim_eta=args.ddim_eta,
            device=device,
            in_channels=C,
            save_results=results_dir,
            max_T_rollout=args.max_T_rollout,
            num_sampling_frame=args.num_sampling_frame,
            num_traj_rollout=args.num_traj_rollout,
            num_eval=args.num_samples,
            batch_size=args.bs,
            run=run,
            extra_note=(
                f"dit_{ms}_{ds}_steps"
                f"{args.samp_diff_steps}"
            ),
            cyclic_runs=cyclic_runs,
        )

        # Ensure previous accelerator operations are finished.
        synchronize_device(device)

        start_time = time.perf_counter()

        out = visualizer.run_all(args.viz_frame_nsp)

        # Wait until all model operations are complete.
        synchronize_device(device)

        inference_time = time.perf_counter() - start_time

        mean_nsp_mse = out["mse_nsp"].mean().item()
        mean_rollout_mse = out["mse_rollout"].mean().item()

        print("NSP prediction shape:", out["pred_nsp"].shape)
        print("Rollout real shape:", out["real_rollout"].shape)
        print(
            "Rollout generated shape:",
            out["gen_rollout"].shape,
        )

        print(f"Mean NSP MSE: {mean_nsp_mse:.10f}")
        print(
            f"Mean rollout MSE: {mean_rollout_mse:.10f}"
        )
        print(
            f"Inference time: {inference_time:.6f} seconds"
        )

        base_name = (
            f"dit_{ms}_{ds}"
            f"_steps{args.samp_diff_steps}"
            f"_run{run_number}"
        )

        tensor_path = os.path.join(
            results_dir,
            f"{base_name}.pt",
        )

        # Save tensors plus the values needed for the table.
        torch.save(
            {
                "pred_nsp": out["pred_nsp"].detach().cpu(),
                "real_rollout": out["real_rollout"].detach().cpu(),
                "gen_rollout": out["gen_rollout"].detach().cpu(),
                "mse_nsp": out["mse_nsp"].detach().cpu(),
                "mse_rollout": out["mse_rollout"].detach().cpu(),
                "mean_nsp_mse": mean_nsp_mse,
                "mean_rollout_mse": mean_rollout_mse,
                "inference_time_seconds": inference_time,
                "model_size": ms,
                "dataset": ds,
                "diffusion_steps": args.samp_diff_steps,
                "ddim_eta": args.ddim_eta,
                "checkpoint": ckpt_path,
                "run": run_number,
            },
            tensor_path,
        )

        table_row = {
            "model": f"BiDiT-{ms}",
            "model_size": ms,
            "dataset": ds,
            "diffusion_steps": args.samp_diff_steps,
            "ddim_eta": args.ddim_eta,
            "run": run_number,
            "forward_nsp_mse": mean_nsp_mse,
            "rollout_mse": mean_rollout_mse,
            "inference_time_seconds": inference_time,
            "num_nsp_samples": args.num_samples,
            "num_rollout_trajectories": args.num_traj_rollout,
            "max_rollout_length": args.max_T_rollout,
            "checkpoint": ckpt_path,
            "tensor_file": tensor_path,
        }

        append_to_csv(
            csv_path=table_csv_path,
            row=table_row,
        )

        print(f"Saved tensors: {tensor_path}")
        print(f"Updated table: {table_csv_path}")


if __name__ == "__main__":
    main()