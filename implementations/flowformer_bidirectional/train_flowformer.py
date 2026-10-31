#%%
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import sys
import torch.nn as nn

from utils.load_data import LoadData
from utils.device_manager import DeviceManager
from utils.vit_flow import ViTTime
from utils.data_config import data_config
from utils.model_config import model_config
from utils.train_args import ArgsConfig
from utils.optimizers import Optimizers
from utils.training_manager import TrainingManager

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.append(project_root)

# Dataset directory stays shared across runs
dataset_dir = os.path.join(project_root, "datasets")
os.makedirs(dataset_dir, exist_ok=True)


def main():
    args = ArgsConfig()
    args_dict = {k: v for k, v in vars(args).items() if k != "parser"}

    print("===== Arguments Used =====")
    for k, v in args_dict.items():
        print(f"{k}: {v}")
    print("==========================")

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx]

    data_configs = data_config()
    var_name = data_configs[args.ds]["var_name"]
    num_traj_use = data_configs[args.ds]["num_traj_use"]
    channels = data_configs[args.ds]["channels"]
    dataset_loc = data_configs[args.ds]["dataset_loc"]

    model_configs = model_config()
    model_dims = model_configs[args.ms]["model_dims"]
    num_heads = model_configs[args.ms]["num_heads"]
    depth = model_configs[args.ms]["depth"]

    model_name = f"flowformer-{args.ms}_{args.ds}"

    if args.tag:
        run_name = f"{model_name}_{args.tag}"
    else:
        run_name = model_name

    # Separate model/results directories for each run
    model_dir = os.path.join(project_root, "models", run_name)
    results_dir = os.path.join(project_root, "results", run_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== Training FlowFormer model with {args.ds} ===")
    print(f"Model name: {model_name}")
    print(f"Run name: {run_name}")
    print(f"Model dir: {model_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Device: {device}")

    loader = LoadData(
        args.ds,
        dataset_loc,
        save_dir=dataset_dir,
        var_name=var_name,
        num_traj_use=num_traj_use,
        channels=channels,
        args=args,
    )

    n_train = data_configs[args.ds]["train_samples"]
    n_val = data_configs[args.ds]["val_samples"]
    n_test = data_configs[args.ds]["test_samples"]

    vids_train, vids_val, vids_test = loader.import_and_split(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    _, _, F, H, W = vids_train.shape

    pH, pW = args.patch_size

    assert H % pH == 0 and W % pW == 0, (
        f"Image size {(H, W)} must be divisible by patch size {args.patch_size}."
    )

    train_dl, val_dl = loader.dataloaders(
        vids_train,
        vids_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    model = ViTTime(
        in_channels=channels,
        time_emb_dim=args.time_emb_dim,
        img_size=(H, W),
        patch_size=args.patch_size,
        embed_dim=model_dims,
        depth=depth,
        num_heads=num_heads,
    )

    print("Model params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    model = model.to(device)

    if args.parallel:
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} CUDA GPUs for DataParallel.")
            model = nn.DataParallel(model)
            bs = int(args.bs // torch.cuda.device_count())
            print(f"Effective batch size with DataParallel: {bs}")
        else:
            print("Ignoring --parallel because multiple CUDA GPUs are not available.")
            bs = args.bs
            print(f"Using single device: {device}. Batch size: {bs}")
    else:
        bs = args.bs
        print(f"Using single device: {device}. Batch size: {bs}")

    opt = Optimizers.get_optimizer(model, args)
    scheduler = Optimizers.get_scheduler(opt, args)

    training_manager = TrainingManager(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=opt,
        scheduler=scheduler,
        device=device,
        args=args,
        args_dict=args_dict,
        model_dir=model_dir,
        results_dir=results_dir,
        model_name=model_name,
    )

    training_manager.run()


if __name__ == "__main__":
    main()