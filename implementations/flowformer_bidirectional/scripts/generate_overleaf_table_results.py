import os
import sys
import argparse
import csv
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.my_model import FlowFormerInference
from utils.device_manager import DeviceManager


def rollout_forward(model, x0, T_roll, num_ode_steps, c_forward):
    N, C, H, W = x0.shape
    traj = torch.zeros(N, T_roll, C, H, W)
    traj[:, 0] = x0.detach().cpu()

    cond = traj[:, 0]

    for t in range(1, T_roll):
        path = model.sample(
            x_cond=cond,
            num_ode_steps=num_ode_steps,
            c=c_forward,
        )
        x_next = path[:, -1].detach().cpu()
        traj[:, t] = x_next
        cond = x_next

    return traj


def rollout_backward(model, xT, T_roll, num_ode_steps, c_backward):
    N, C, H, W = xT.shape
    traj = torch.zeros(N, T_roll, C, H, W)
    traj[:, 0] = xT.detach().cpu()

    cond = traj[:, 0]

    for t in range(1, T_roll):
        path = model.sample(
            x_cond=cond,
            num_ode_steps=num_ode_steps,
            c=c_backward,
        )
        x_prev = path[:, -1].detach().cpu()
        traj[:, t] = x_prev
        cond = x_prev

    return traj


def rollout_error_per_traj(pred, true):
    """
    pred, true: (N, T, C, H, W)
    returns: (N,)
    """
    return ((pred - true) ** 2).mean(dim=(1, 2, 3, 4))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="datasets/vids_test_ce-rm.npy")
    parser.add_argument("--num_ode_steps", type=int, default=5)
    parser.add_argument("--max_T_rollout", type=int, default=20)
    parser.add_argument("--K_forward", type=int, default=16)
    parser.add_argument("--K_cyclic", type=int, default=8)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--traj_indices", type=int, nargs="*", default=None)
    parser.add_argument("--all_trajs", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    dataset_path = args.dataset
    if not os.path.isfile(dataset_path):
        dataset_path = os.path.join(PROJECT_ROOT, dataset_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["args"]

    ms = config["ms"]
    ds = config["ds"]

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    vids_test = torch.from_numpy(np.load(dataset_path)).float()
    N_total, T_total, C, H, W = vids_test.shape

    if args.all_trajs:
        traj_indices = list(range(N_total))
    elif args.traj_indices is None:
        traj_indices = [0, N_total // 2, N_total - 1]
    else:
        traj_indices = args.traj_indices

    vids = vids_test[traj_indices]
    T_roll = min(args.max_T_rollout, vids.shape[1])

    real_forward = vids[:, :T_roll].detach().cpu()

    c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)
    c_backward = torch.tensor([0.0, 1.0], dtype=torch.float32)

    print("===================================")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Model size: {ms}")
    print(f"Dataset name: {ds}")
    print(f"Selected trajectory indices: {traj_indices}")
    print(f"T_roll: {T_roll}")
    print(f"K_forward: {args.K_forward}")
    print(f"K_cyclic: {args.K_cyclic}")
    print("===================================")

    model = FlowFormerInference.from_pretrained(
        checkpoint_path=ckpt_path,
        img_size=(H, W),
        device=device,
    )

    # -------------------------------------------------------
    # Forward K stochastic rollouts
    # -------------------------------------------------------
    forward_errors = []

    for k in range(args.K_forward):
        print(f"Forward candidate {k + 1}/{args.K_forward}")

        gen_forward = rollout_forward(
            model=model,
            x0=real_forward[:, 0],
            T_roll=T_roll,
            num_ode_steps=args.num_ode_steps,
            c_forward=c_forward,
        )

        err = rollout_error_per_traj(gen_forward, real_forward)
        forward_errors.append(err)

    forward_errors = torch.stack(forward_errors, dim=0)
    forward_mean = forward_errors.mean(dim=0)
    forward_std = forward_errors.std(dim=0, unbiased=False)

    # -------------------------------------------------------
    # Cyclic K candidates
    # -------------------------------------------------------
    cyclic_scores = []
    cyclic_true_errors = []

    for k in range(args.K_cyclic):
        print(f"Cyclic candidate {k + 1}/{args.K_cyclic}")

        fwd_traj = rollout_forward(
            model=model,
            x0=real_forward[:, 0],
            T_roll=T_roll,
            num_ode_steps=args.num_ode_steps,
            c_forward=c_forward,
        )

        bwd_traj = rollout_backward(
            model=model,
            xT=fwd_traj[:, -1],
            T_roll=T_roll,
            num_ode_steps=args.num_ode_steps,
            c_backward=c_backward,
        )

        fwd_reversed = torch.flip(fwd_traj, dims=[1])

        path_cyclic_error = ((bwd_traj - fwd_reversed) ** 2).mean(dim=(2, 3, 4))
        path_cyclic_score = path_cyclic_error.mean(dim=1)

        true_err = rollout_error_per_traj(fwd_traj, real_forward)

        cyclic_scores.append(path_cyclic_score)
        cyclic_true_errors.append(true_err)

    cyclic_scores = torch.stack(cyclic_scores, dim=0)
    cyclic_true_errors = torch.stack(cyclic_true_errors, dim=0)

    best_cyclic_idx = torch.argmin(cyclic_scores, dim=0)
    best_cyclic_error = torch.zeros(len(traj_indices))

    for i in range(len(traj_indices)):
        best_cyclic_error[i] = cyclic_true_errors[best_cyclic_idx[i], i]

    # -------------------------------------------------------
    # Dataset-level averages
    # -------------------------------------------------------
    dataset_forward_mean = forward_mean.mean()
    dataset_forward_std = forward_mean.std(unbiased=False)

    dataset_cyclic_mean = best_cyclic_error.mean()
    dataset_cyclic_std = best_cyclic_error.std(unbiased=False)

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    tag = f"flowformer_{ms}_{ds}"

    suffix = "all_trajs" if args.all_trajs else "selected_trajs"

    pt_path = os.path.join(args.out_dir, f"overleaf_table_{tag}_{suffix}.pt")
    txt_path = os.path.join(args.out_dir, f"overleaf_table_{tag}_{suffix}.txt")
    csv_path = os.path.join(args.out_dir, f"overleaf_table_{tag}_{suffix}.csv")

    torch.save(
        {
            "traj_indices": traj_indices,
            "forward_errors": forward_errors,
            "forward_mean": forward_mean,
            "forward_std": forward_std,
            "cyclic_scores": cyclic_scores,
            "cyclic_true_errors": cyclic_true_errors,
            "best_cyclic_idx": best_cyclic_idx,
            "best_cyclic_error": best_cyclic_error,
            "dataset_forward_mean": dataset_forward_mean,
            "dataset_forward_std": dataset_forward_std,
            "dataset_cyclic_mean": dataset_cyclic_mean,
            "dataset_cyclic_std": dataset_cyclic_std,
            "K_forward": args.K_forward,
            "K_cyclic": args.K_cyclic,
            "T_roll": T_roll,
            "metric": "trajectory_mse_mean_over_T_C_H_W",
        },
        pt_path,
    )

    with open(txt_path, "w") as f:
        f.write(f"Model: {tag}\n")
        f.write("Metric: trajectory MSE averaged over T,C,H,W\n")
        f.write(f"K_forward: {args.K_forward}\n")
        f.write(f"K_cyclic: {args.K_cyclic}\n")
        f.write(f"Trajectory indices: {traj_indices}\n\n")

        f.write("Forward row: E_for,K=16 = mean ± std\n")
        for i, traj_idx in enumerate(traj_indices):
            f.write(
                f"tau_n={traj_idx}: "
                f"{forward_mean[i].item():.6f} ± {forward_std[i].item():.6f}\n"
            )

        f.write("\nCyclic row: E_best_cyclic,K=8\n")
        for i, traj_idx in enumerate(traj_indices):
            f.write(
                f"tau_n={traj_idx}: "
                f"{best_cyclic_error[i].item():.6f} "
                f"(best candidate {best_cyclic_idx[i].item()})\n"
            )

        f.write("\nDataset-level averages:\n")
        f.write(
            f"Forward: {dataset_forward_mean.item():.6f} "
            f"± {dataset_forward_std.item():.6f}\n"
        )
        f.write(
            f"Cyclic: {dataset_cyclic_mean.item():.6f} "
            f"± {dataset_cyclic_std.item():.6f}\n"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "traj_idx",
                "forward_mean",
                "forward_std",
                "best_cyclic_error",
                "best_cyclic_idx",
            ]
        )

        for i, traj_idx in enumerate(traj_indices):
            writer.writerow(
                [
                    tag,
                    traj_idx,
                    forward_mean[i].item(),
                    forward_std[i].item(),
                    best_cyclic_error[i].item(),
                    best_cyclic_idx[i].item(),
                ]
            )

        writer.writerow([])
        writer.writerow(
            [
                "dataset_level",
                "all_selected_trajs",
                dataset_forward_mean.item(),
                dataset_forward_std.item(),
                dataset_cyclic_mean.item(),
                dataset_cyclic_std.item(),
            ]
        )

    print("\nSaved:")
    print(pt_path)
    print(txt_path)
    print(csv_path)

    print("\n===== TABLE VALUES =====")
    print("Forward row:")
    for i, traj_idx in enumerate(traj_indices):
        print(
            f"tau_n={traj_idx}: "
            f"{forward_mean[i].item():.6f} ± {forward_std[i].item():.6f}"
        )

    print("\nCyclic row:")
    for i, traj_idx in enumerate(traj_indices):
        print(
            f"tau_n={traj_idx}: "
            f"{best_cyclic_error[i].item():.6f} "
            f"(best candidate {best_cyclic_idx[i].item()})"
        )

    print("\nDataset-level averages:")
    print(
        f"Forward: {dataset_forward_mean.item():.6f} "
        f"± {dataset_forward_std.item():.6f}"
    )
    print(
        f"Cyclic: {dataset_cyclic_mean.item():.6f} "
        f"± {dataset_cyclic_std.item():.6f}"
    )


if __name__ == "__main__":
    main()