import os
import sys
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

    models_dir = os.path.join(project_root, "models")
    ckpt_path = os.path.join(models_dir, ckpt)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path


def main():
    args = InferArgsConfig()

    # Backward compatible fallback.
    # For command-line support, add --cyclic_runs in utils/infer_args.py.
    cyclic_runs = getattr(args, "cyclic_runs", [1])

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]
    print(f"Using device: {device}")

    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    
    dataset_root = args.dataset_root if args.dataset_root else project_root
    dataset_dir = os.path.join(dataset_root, "datasets")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.ckpt)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["args"]

    ds = config["ds"]
    ms = config["ms"]
    dit_arch = config.get("dit_arch", "legacy")

    test_path = os.path.join(dataset_dir, f"vids_test_{ds}.npy")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    test_data = np.load(test_path)
    vids_test = torch.from_numpy(test_data).float()

    N, T, C, H, W = vids_test.shape
    img_size = (H, W)

    print(f"Test videos shape: {vids_test.shape}")
    print(f"Image size: {img_size}")
    print(f"Cyclic candidate runs: {cyclic_runs}")

    model = DiTInference.from_pretrained(
        checkpoint_path=ckpt_path,
        img_size=img_size,
        device=device,
    )

    for run in range(args.num_runs):
        print(f"\n=== Run {run + 1}/{args.num_runs} ===")

        visualizer = TestRolloutVisualizer(
            model=model,
            vids_test=vids_test,
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
                f"dit_{ms}_{ds}"
                if dit_arch == "legacy"
                else f"dit_{dit_arch}_{ms}_{ds}"
            ),
            cyclic_runs=cyclic_runs,
        )

        out = visualizer.run_all(args.viz_frame_nsp)

        print("NSP prediction shape:", out["pred_nsp"].shape)
        print("Rollout real shape:", out["real_rollout"].shape)
        print("Rollout generated shape:", out["gen_rollout"].shape)
        print("Mean NSP MSE:", out["mse_nsp"].mean().item())
        print("Mean rollout MSE:", out["mse_rollout"].mean().item())


if __name__ == "__main__":
    main()
