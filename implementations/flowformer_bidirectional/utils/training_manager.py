import os
import time

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.flow_trainer import Trainer

class TrainingManager:
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        optimizer,
        scheduler,
        device,
        args,
        args_dict,
        model_dir,
        results_dir,
        model_name,
    ):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.args_dict = args_dict
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.model_name = model_name

        self.start_epoch = 0

        self.train_loss_list = []
        self.train_flow_loss_list = []
        self.train_cycle_loss_list = []

        self.val_loss_list = []
        self.val_nsp_loss_list = []

    def resume_if_needed(self):
        if not self.args.resume:
            return

        if self.args.ckpt is None:
            raise ValueError("--resume requires --ckpt <CHECKPOINT>")

        resume_path = os.path.join(self.model_dir, self.args.ckpt)

        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=self.device)

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint missing model_state_dict/state_dict")

        state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target_model.load_state_dict(state_dict, strict=True)

        if ckpt.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.start_epoch = ckpt.get("epoch", 0)

        self.train_loss_list = ckpt.get("train_loss_list", [])
        self.train_flow_loss_list = ckpt.get("train_flow_loss_list", [])
        self.train_cycle_loss_list = ckpt.get("train_cycle_loss_list", [])

        self.val_loss_list = ckpt.get("val_loss_list", [])
        self.val_nsp_loss_list = ckpt.get("val_nsp_loss_list", [])

        print(f"Resumed from {resume_path}")
        print(f"Starting from epoch {self.start_epoch + 1}")

    def run(self):
        self.resume_if_needed()

        if self.start_epoch >= self.args.epochs:
            print(
                f"Checkpoint already reached epoch {self.start_epoch}. "
                f"args.epochs={self.args.epochs}, so no further training will run."
            )
            self.plot_losses()
            return self.train_loss_list, self.val_loss_list

        begin_time = time.time()

        for ep in range(self.start_epoch + 1, self.args.epochs + 1):

            tr_loss = Trainer.train_epoch(
                model=self.model,
                dataloader=self.train_dl,
                optimizer=self.optimizer,
                device=self.device,
            )

            val_loss = Trainer.val_epoch(
                model=self.model,
                dataloader=self.val_dl,
                device=self.device,
            )

            val_nsp_loss = Trainer.nsp_epoch(
                model=self.model,
                dataloader=self.val_dl,
                num_ode_steps=self.args.num_ode_steps,
                device=self.device,
                max_batches=1, # for faster evaluation during training (use 1 batch)
            )

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - begin_time

            print(
                f"Epoch {ep:03d} | "
                f"Time: {elapsed/60:.2f} min | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss (flow): {tr_loss:.4f} | "
                f"Val Loss (flow): {val_loss:.4f} | "
                f"Val Loss (NSP): {val_nsp_loss:.4f}"
            )

            self.train_loss_list.append(tr_loss)
            self.val_loss_list.append(val_loss)
            self.val_nsp_loss_list.append(val_nsp_loss)

            self.save_checkpoint(ep)

        self.plot_losses()


    def save_checkpoint(self, epoch):
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "train_loss_list": self.train_loss_list,
            "train_flow_loss_list": self.train_flow_loss_list,
            "train_cycle_loss_list": self.train_cycle_loss_list,
            "val_loss_list": self.val_loss_list,
            "val_nsp_loss_list": self.val_nsp_loss_list,
            "args": self.args_dict,
        }

        ckpt_path = os.path.join(self.model_dir, f"{self.model_name}_{epoch}.pth")
        torch.save(checkpoint, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_list, label="Train Total Loss")
        plt.plot(self.train_flow_loss_list, label="Train Flow Loss")
        plt.plot(self.val_loss_list, label="Val Flow Loss")

        if len(self.train_cycle_loss_list) > 0:
            plt.plot(self.train_cycle_loss_list, label="Train Cycle Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("FlowFormer Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_flow_cycle_loss_curves.png"))
        plt.close()

        if len(self.val_nsp_loss_list) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.val_nsp_loss_list, label="Val NSP MSE")
            plt.xlabel("Epoch")
            plt.ylabel("State MSE")
            plt.title("Next-Step Prediction Validation MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_nsp_mse_curve.png"))
            plt.close()