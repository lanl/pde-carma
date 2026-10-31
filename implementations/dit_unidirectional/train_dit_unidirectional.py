"""Train the current DiT architecture on forward pairs only.

The model architecture is intentionally the same as the bidirectional trainer.
The only training difference is that every adjacent pair is x_t -> x_{t+1}
with the fixed forward control c=[1, 0].  This makes Uni-Fwd vs Bi-Fwd an
architecture-matched comparison.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.append(project_root)

from utils.data_config import data_config
from utils.device_manager import DeviceManager
from utils.load_data import LoadData
from utils.model_config import model_config
from utils.optimizers import Optimizers
from utils.train_args import ArgsConfig
from utils.training_manager import TrainingManager
from utils.vit_diffusion import ViTTime


class ForwardNextStepPairs(Dataset):
    """Turn videos [N,T,C,H,W] into forward-only adjacent-state pairs."""

    def __init__(self, videos_5d: torch.Tensor):
        if videos_5d.ndim != 5:
            raise ValueError(
                f"Expected videos with shape [N,T,C,H,W], got {videos_5d.shape}"
            )

        self.videos = videos_5d
        self.N, self.T, _, _, _ = videos_5d.shape
        self.index = [
            (n, t)
            for n in range(self.N)
            for t in range(self.T - 1)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        n, t = self.index[idx]
        x_cond = self.videos[n, t]
        x_target = self.videos[n, t + 1]
        c_forward = torch.tensor([1.0, 0.0], dtype=torch.float32)
        return x_cond.float(), x_target.float(), c_forward


class ForwardLoadData(LoadData):
    """Reuse current loading/normalization, but build only forward pairs."""

    def dataloaders(self, vids_train, vids_val, batch_size=64, num_workers=0):
        train_ds = ForwardNextStepPairs(vids_train)
        val_ds = ForwardNextStepPairs(vids_val)

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        if len(train_dl) == 0:
            raise RuntimeError(
                "Training DataLoader is empty. Reduce --bs or check the split."
            )
        if len(val_dl) == 0:
            raise RuntimeError("Validation DataLoader is empty.")

        print(f"Len DataLoaders: {len(train_dl)}, {len(val_dl)}")
        print("Unidirectional mode: forward c=[1,0] only")
        return train_dl, val_dl


def main():
    args = ArgsConfig()
    args.patch_size = tuple(args.patch_size)

    # Saved in the checkpoint so inference/results scripts can identify the run.
    args.direction = "forward"
    args.model_type = "unidirectional_dit"

    args_dict = {
        key: value
        for key, value in vars(args).items()
        if key != "parser"
    }

    print("===== Arguments Used =====")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("==========================")

    devices = DeviceManager.list_devices()
    if args.device_idx < 0 or args.device_idx >= len(devices):
        raise IndexError(
            f"device_idx={args.device_idx}, but available devices are "
            f"0..{len(devices) - 1}."
        )
    device = devices[args.device_idx]

    dataset_root = args.dataset_root if args.dataset_root else project_root
    dataset_dir = os.path.join(dataset_root, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    data_configs = data_config(dataset_root)
    dataset_cfg = data_configs[args.ds]

    model_configs = model_config()
    model_cfg = model_configs[args.ms]

    if args.dit_arch == "legacy":
        base_name = f"dit-{args.ms}_{args.ds}"
    else:
        base_name = f"dit-{args.dit_arch}-{args.ms}_{args.ds}"

    model_name = f"{base_name}_forward_unidirectional"
    run_name = f"{model_name}_{args.tag}" if args.tag else model_name

    model_dir = os.path.join(project_root, "models", run_name)
    results_dir = os.path.join(project_root, "results", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Training forward-only DiT on {args.ds} ===")
    print(f"Model name:  {model_name}")
    print(f"Run name:    {run_name}")
    print(f"Model dir:   {model_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Device:      {device}")

    loader = ForwardLoadData(
        args.ds,
        dataset_cfg["dataset_loc"],
        save_dir=dataset_dir,
        var_name=dataset_cfg["var_name"],
        num_traj_use=dataset_cfg["num_traj_use"],
        channels=dataset_cfg["channels"],
        args=args,
    )

    vids_train, vids_val, _ = loader.import_and_split(
        n_train=dataset_cfg["train_samples"],
        n_val=dataset_cfg["val_samples"],
        n_test=dataset_cfg["test_samples"],
    )

    _, _, _, height, width = vids_train.shape
    patch_h, patch_w = args.patch_size
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by "
            f"patch size {args.patch_size}."
        )

    train_dl, val_dl = loader.dataloaders(
        vids_train,
        vids_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    # Keep exactly the same architecture as the current bidirectional branch.
    # Only the training pairs differ; the direction embedding is retained and
    # always receives c=[1,0].
    model = ViTTime(
        in_channels=dataset_cfg["channels"],
        time_emb_dim=args.time_emb_dim,
        img_size=(height, width),
        patch_size=args.patch_size,
        embed_dim=model_cfg["model_dims"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        dit_arch=args.dit_arch,
    )

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"DiT architecture: {args.dit_arch}")
    print(f"Model params: {num_params / 1e6:.4f} M")
    print("Direction-control MLP: retained; training uses c=[1,0] only")

    model = model.to(device)

    if args.parallel:
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            print(
                f"Using {torch.cuda.device_count()} CUDA GPUs with DataParallel."
            )
            model = nn.DataParallel(model)
        else:
            print(
                "Ignoring --parallel because multiple CUDA GPUs are not available."
            )

    optimizer = Optimizers.get_optimizer(model, args)
    scheduler = Optimizers.get_scheduler(optimizer, args)

    manager = TrainingManager(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        args_dict=args_dict,
        model_dir=model_dir,
        results_dir=results_dir,
        model_name=model_name,
    )
    manager.run()


if __name__ == "__main__":
    main()
