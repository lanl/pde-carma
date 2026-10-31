import torch
import numpy as np
import os
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt

class Testing():
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name
        
    # ---------- helpers ----------
    def _to_numpy_img(self, t: torch.Tensor, field_idx: int = 0):
        """
        t: (F,H,W) or (H,W). Returns 2D numpy image for plotting.
        If F>1, pick channel 'field_idx'.
        """
        if t.ndim == 3:
            t = t[field_idx]
        return t.detach().cpu().float().numpy()
    
    def _mse_map(self, pred: torch.Tensor, true: torch.Tensor):
        """
        pred/true: (F,H,W) tensors. Returns per-pixel MSE averaged over F -> (H,W) numpy.
        """
        err = (pred - true) ** 2
        if err.ndim == 3:
            err = err.mean(dim=0)  # mean across F
        return err.detach().cpu().float().numpy()
    
    def _mae_scalar(self, pred: torch.Tensor, true: torch.Tensor):
        return torch.mean(torch.abs(pred - true)).item()
    
    def _mse_scalar(self, pred: torch.Tensor, true: torch.Tensor):
        return torch.mean((pred - true) ** 2).item()
    
    # ---------- 1) Single-step predictions on test set ----------
    def save_nextstep_examples(self, vids_test, out_dir, max_samples=16, field_idx=0, device=None):
        os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        device = device or next(self.model.parameters()).device
    
        N, T, F, H, W = vids_test.shape
        count = 0
        with torch.no_grad():
            for n in range(N):
                for t in range(T - 1):
                    x_prev = vids_test[n, t].unsqueeze(0).to(device)   # (1,F,H,W)
                    x_next = vids_test[n, t + 1].to(device)            # (F,H,W)
    
                    pred = self.model(x_prev)[0]                             # (F,H,W)
    
                    # Build panel
                    true_img = self._to_numpy_img(x_next, field_idx)
                    pred_img = self._to_numpy_img(pred.clamp(0, 1), field_idx)   # clamp for visualization
                    mse_img  = self._mse_map(pred, x_next)
    
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                    ax[0].imshow(true_img, cmap="plasma"); ax[0].set_title("TRUE"); ax[0].axis("off")
                    ax[1].imshow(pred_img, cmap="plasma"); ax[1].set_title("PRED"); ax[1].axis("off")
                    im = ax[2].imshow(mse_img, cmap="plasma"); ax[2].set_title("MSE (per-pixel)"); ax[2].axis("off")
                    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
                    fig.suptitle(self.model_name, fontsize=18)
                    
                    fname = os.path.join(out_dir, 
                            f"{self.model_name}_nextstep_field{field_idx}_n{n:03d}_t{t:03d}.png")
                    plt.tight_layout()
                    plt.savefig(fname, dpi=150)
                    plt.close(fig)
    
                    count += 1
                    if count >= max_samples:
                        return
    
    # ---------- 2) Full rollout from first frame (20 steps) ----------
    def save_rollout_panel(self, vids_test, out_dir, sample_index=0, 
                           horizon=20, field_idx=0, device=None):
        """
        Saves a 3-row panel:
          row1 = TRUE frames,
          row2 = PRED frames (from autoregressive rollout),
          row3 = MSE heatmaps (title shows MSE & MAE).
        """
        os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        device = device or next(self.model.parameters()).device
    
        N, T, F, H, W = vids_test.shape
        k = min(horizon, T - 1)
    
        with torch.no_grad():
            # init buffer with the first time step
            prev = vids_test[sample_index, 0].unsqueeze(0).to(device)  # (1,F,H,W)
    
            true_frames = []
            pred_frames = []
            mse_maps    = []
            mse_vals    = []
            mae_vals    = []
    
            for t in range(k):
                # ground truth next frame
                gt = vids_test[sample_index, t + 1].to(device)         # (F,H,W)
    
                # predict next
                pred = self.model(prev)[0]                                   # (F,H,W)
    
                # collect visuals/metrics
                true_frames.append(self._to_numpy_img(gt, field_idx))
                pred_frames.append(self._to_numpy_img(pred.clamp(0, 1), field_idx))
                mse_map = self._mse_map(pred, gt)
                mse_maps.append(mse_map)
                mse_scalar = self._mse_scalar(pred, gt)
                mse_vals.append(mse_scalar)
                mae_scalar = self._mae_scalar(pred, gt)
                mae_vals.append(mae_scalar)
    
                # update prev for next step (autoregressive)
                prev = pred.unsqueeze(0)
    
            # Plot 3-row panel
            fig, ax = plt.subplots(3, k, figsize=(2.5*k, 6))
            if k == 1:  # keep indexing simple for k=1
                ax = np.array([[ax[0]], [ax[1]], [ax[2]]])
    
            for t in range(k):
                # Row 1: TRUE
                ax[0, t].imshow(true_frames[t], cmap="plasma")
                ax[0, t].set_title(f"t={t+1} TRUE"); ax[0, t].axis("off")
    
                # Row 2: PRED
                ax[1, t].imshow(pred_frames[t], cmap="plasma")
                ax[1, t].set_title(f"t={t+1} PRED"); ax[1, t].axis("off")
    
                # Row 3: MSE heatmap with metrics in title
                im = ax[2, t].imshow(mse_maps[t], cmap="plasma")
                ax[2, t].set_title(f"MAE={mae_vals[t]:.4e}",
                                   fontsize = 8)
                ax[2, t].axis("off")
    
            # single colorbar for the last row
            cbar_ax = fig.add_axes([0.92, 0.10, 0.015, 0.25])
            fig.colorbar(im, cax=cbar_ax, location="right")
            fig.suptitle(self.model_name, fontsize=18)
            
            #plt.tight_layout()
    
            fname = os.path.join(out_dir, 
                    f"{self.model_name}_rollout_sample{sample_index:03d}_field{field_idx}_k{k}.png")
            plt.savefig(fname, dpi=150)
            plt.close(fig)