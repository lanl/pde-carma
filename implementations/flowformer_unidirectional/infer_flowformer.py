import os
import sys
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.append(project_root)

from utils.my_model import FlowFormerInference
from utils.infer_args import InferArgsConfig
from utils.device_manager import DeviceManager
from utils.get_results import TestRolloutVisualizer


def resolve_checkpoint_path(ckpt):
    if os.path.isfile(ckpt):
        return ckpt

    models_dir = os.path.join(project_root, "models")
    ckpt_path = os.path.join(models_dir, ckpt)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


def resolve_test_data_path(dataset_dir, ds, direction):
    """
    Prefer direction-specific test files:
        vids_test_<ds>_<direction>.npy

    Fall back to old format only if direction-specific file is missing.
    """

    direction_test_path = os.path.join(
        dataset_dir,
        f"vids_test_{ds}_{direction}.npy",
    )

    old_test_path = os.path.join(
        dataset_dir,
        f"vids_test_{ds}.npy",
    )

    if os.path.exists(direction_test_path):
        return direction_test_path

    if os.path.exists(old_test_path):
        print(
            f"WARNING: Direction-specific test file not found:\n"
            f"  {direction_test_path}\n"
            f"Falling back to old test file:\n"
            f"  {old_test_path}\n"
            f"For backward models, this may be incorrect."
        )
        return old_test_path

    raise FileNotFoundError(
        f"Test data not found. Tried:\n"
        f"  {direction_test_path}\n"
        f"  {old_test_path}"
    )


def main():
    args = InferArgsConfig()

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]
    print(f"Using device: {device}")

    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    dataset_dir = os.path.join(project_root, "datasets")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.ckpt)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["args"]

    ds = config["ds"]
    ms = config["ms"]
    direction = config.get("direction", "forward")

    if direction not in ["forward", "backward"]:
        raise ValueError(
            f"Unsupported checkpoint direction: {direction}. "
            "Expected 'forward' or 'backward'."
        )

    print("===== Inference Config =====")
    print(f"Model size: {ms}")
    print(f"Dataset: {ds}")
    print(f"Direction: {direction}")
    print(f"Checkpoint: {ckpt_path}")
    print("============================")

    test_path = resolve_test_data_path(dataset_dir, ds, direction)
    print(f"Loading test data from: {test_path}")

    test_data = np.load(test_path)
    vids_test = torch.from_numpy(test_data).float()

    # vids_test: [N, T, C, H, W]
    N, T, C, H, W = vids_test.shape
    img_size = (H, W)

    print(f"Test videos shape: {vids_test.shape}")
    print(f"Image size: {img_size}")

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=ckpt_path,
        img_size=img_size,
        device=device,
    )

    for run in range(args.num_runs):
        print(f"\n=== Run {run + 1}/{args.num_runs} ===")

        visualizer = TestRolloutVisualizer(
            model=model,
            vids_test=vids_test,
            num_ode_steps=args.num_ode_steps,
            device=device,
            in_channels=C,
            save_results=results_dir,
            max_T_rollout=args.max_T_rollout,
            num_sampling_frame=args.num_sampling_frame,
            num_traj_rollout=args.num_traj_rollout,
            num_eval=args.num_samples,
            batch_size=args.bs,
            run=run,
            extra_note=f"flowformer_{ms}_{ds}_{direction}",
            direction=direction,
        )

        out = visualizer.run_all(args.viz_frame_nsp)

        print("NSP prediction shape:", out["pred_nsp"].shape)
        print("Rollout real shape:", out["real_rollout"].shape)
        print("Rollout generated shape:", out["gen_rollout"].shape)
        print("Mean NSP MSE:", out["mse_nsp"].mean().item())
        print("Mean rollout MSE:", out["mse_rollout"].mean().item())


if __name__ == "__main__":
    main()