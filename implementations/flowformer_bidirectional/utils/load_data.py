import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data_arr import data_arr_nc, data_arr_npy


class NextStepPairs(Dataset):
    """
    Turns a (N,T,F,H,W) tensor into bidirectional pairs.

    Forward example:
        x_t     -> x_{t+1}, c = [1, 0]

    Backward example:
        x_{t+1} -> x_t,     c = [0, 1]
    """

    def __init__(self, videos_5d: torch.Tensor):
        assert videos_5d.ndim == 5, "Expected (N,T,F,H,W)"

        self.X = videos_5d
        self.N, self.T, self.F, self.H, self.W = videos_5d.shape

        self.index = []
        for n in range(self.N):
            for t in range(self.T - 1):
                self.index.append((n, t, 0))  # forward
                self.index.append((n, t, 1))  # backward

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t, mode = self.index[i]

        if mode == 0:
            x_cond = self.X[n, t]
            x_next = self.X[n, t + 1]
            c = torch.tensor([1.0, 0.0], dtype=torch.float32)
        else:
            x_cond = self.X[n, t + 1]
            x_next = self.X[n, t]
            c = torch.tensor([0.0, 1.0], dtype=torch.float32)

        return x_cond.float(), x_next.float(), c


class LoadData:
    def __init__(
        self,
        dataset_name,
        dataset_loc,
        save_dir,
        var_name,
        num_traj_use=None,
        channels=4,
        args=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_loc = dataset_loc
        self.save_dir = save_dir
        self.var_name = var_name
        self.num_traj_use = num_traj_use
        self.channels = channels
        self.args = args

    def import_and_split(self, n_train=1030, n_val=100, n_test=130):

        # ---------------------------------------------------------
        # Pre-split NPY datasets: RD2D, SWE, Active Matter
        # ---------------------------------------------------------
        if self.dataset_loc.endswith(".npy") and "vids_train_" in os.path.basename(self.dataset_loc):
            train_path = self.dataset_loc

            val_path = train_path.replace("vids_train_", "vids_val_")
            test_path = train_path.replace("vids_train_", "vids_test_")

            print(f"Loading train: {train_path}")
            print(f"Loading val:   {val_path}")
            print(f"Loading test:  {test_path}")

            vids_train = torch.from_numpy(np.load(train_path)).float()
            vids_val = torch.from_numpy(np.load(val_path)).float()
            vids_test = torch.from_numpy(np.load(test_path)).float()

            vids_train = vids_train[:, :, :self.channels]
            vids_val = vids_val[:, :, :self.channels]
            vids_test = vids_test[:, :, :self.channels]

            # normalization statistics from TRAIN ONLY
            mu_ch = vids_train.mean(dim=(0, 1, 3, 4), keepdim=True,)
            std_ch = vids_train.std(dim=(0, 1, 3, 4), keepdim=True,) + 1e-6

            vids_train.sub_(mu_ch).div_(std_ch)
            vids_val.sub_(mu_ch).div_(std_ch)
            vids_test.sub_(mu_ch).div_(std_ch)

        # ---------------------------------------------------------
        # NS-PwC, CE-RM, CE-RPUI: Load and split from single NC or NPY file
        # ---------------------------------------------------------
        else:
            if self.dataset_loc.endswith(".nc"):
                arr = data_arr_nc(self.dataset_loc, self.var_name,)
            elif self.dataset_loc.endswith(".npy"):
                arr = data_arr_npy(self.dataset_loc)
            else:
                raise ValueError(f"Unsupported dataset format: {self.dataset_loc}")

            arr = np.nan_to_num(arr)

            vids = torch.from_numpy(arr)

            print("Loaded vids shape:", tuple(vids.shape),)

            if self.num_traj_use is not None:
                vids = vids[:self.num_traj_use]

            vids = vids[:, :, :self.channels]

            mu_ch = vids.mean(dim=(0, 1, 3, 4), keepdim=True,)
            std_ch = vids.std(dim=(0, 1, 3, 4), keepdim=True,) + 1e-6

            vids = (vids - mu_ch) / std_ch

            vids_train = vids[:n_train]

            vids_val = vids[n_train:n_train + n_val]

            vids_test = vids[n_train + n_val:n_train + n_val + n_test]

        print(f"Train set: {vids_train.shape}, "
              f"Val set: {vids_val.shape}, "
              f"Test set: {vids_test.shape}")

        os.makedirs(self.save_dir, exist_ok=True)

        val_vids_path = os.path.join(self.save_dir,
            f"vids_val_{self.dataset_name}.npy",)
        np.save(val_vids_path, vids_val.cpu().numpy())

        test_vids_path = os.path.join(self.save_dir,
            f"vids_test_{self.dataset_name}.npy",)
        np.save(test_vids_path, vids_test.cpu().numpy())

        print(f"Saved val set to {val_vids_path} \n"
            f"and test set to {test_vids_path}.")

        return vids_train, vids_val, vids_test

    def dataloaders(self, vids_train, vids_val, batch_size=64, num_workers=0):
        train_ds = NextStepPairs(vids_train)
        val_ds = NextStepPairs(vids_val)

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

        assert len(train_dl) > 0, (
            "train_dl is empty. Reduce batch_size or check dataset split."
        )
        assert len(val_dl) > 0, "val_dl is empty. Check validation split."

        print(f"Len DataLoaders: {len(train_dl)}, {len(val_dl)}")
        print("Bidirectional mode enabled: forward c=[1,0], backward c=[0,1]")

        return train_dl, val_dl