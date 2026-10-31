import os
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np
import matplotlib.pyplot as plt

def visualize_real_vs_generated(
    real_batch,
    gen_batch,
    channel=0,
    num_to_show=4,
    saveloc=None,
    frame_no=None,
    extra_note="",
):
    real_batch = real_batch.detach().cpu()
    gen_batch = gen_batch.detach().cpu()

    assert real_batch.ndim == 4, f"Expected real_batch [B,C,H,W], got {real_batch.shape}"
    assert gen_batch.ndim == 4, f"Expected gen_batch [B,C,H,W], got {gen_batch.shape}"
    assert real_batch.shape[1] == gen_batch.shape[1], "Channel count mismatch"

    B, C, H, W = real_batch.shape

    assert 0 <= channel < C, f"channel {channel} out of range 0..{C-1}"

    num_to_show = min(num_to_show, real_batch.size(0), gen_batch.size(0))

    if num_to_show <= 0:
        raise ValueError(f"num_to_show must be positive, got {num_to_show}")

    real_chan = real_batch[:num_to_show, channel]
    gen_chan = gen_batch[:num_to_show, channel]

    fig, axes = plt.subplots(2, num_to_show, figsize=(3 * num_to_show, 6))

    if num_to_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(num_to_show):
        img_r = real_chan[i]
        r_min, r_max = img_r.min(), img_r.max()
        img_r = (img_r - r_min) / (r_max - r_min + 1e-6)

        ax = axes[0, i]
        ax.imshow(img_r.numpy(), cmap="viridis")
        ax.set_title(f"Real #{i}, ch={channel}")
        ax.axis("off")

        img_g = gen_chan[i]
        g_min, g_max = img_g.min(), img_g.max()
        img_g = (img_g - g_min) / (g_max - g_min + 1e-6)

        ax = axes[1, i]
        ax.imshow(img_g.numpy(), cmap="viridis")
        ax.set_title(f"Gen #{i}, ch={channel}")
        ax.axis("off")

    plt.tight_layout()

    if saveloc is not None:
        os.makedirs(saveloc, exist_ok=True)
        save_path = os.path.join(
            saveloc,
            f"real_vs_gen_frame{frame_no}_ch{channel}_{extra_note}.png",
        )
        plt.savefig(save_path, dpi=300)

    plt.close()


def visualize_full_trajectory(
    real_traj,
    gen_traj,
    seq_idx=0,
    channel=0,
    max_time_steps=8,
    saveloc=None,
    extra_note="",
):
    real_traj = real_traj.detach().cpu()
    gen_traj = gen_traj.detach().cpu()

    assert real_traj.ndim == 5, f"Expected real_traj [N,T,C,H,W], got {real_traj.shape}"
    assert gen_traj.shape == real_traj.shape, "real_traj and gen_traj must have same shape"

    N, T, C, H, W = real_traj.shape

    assert 0 <= seq_idx < N, f"seq_idx {seq_idx} out of range 0..{N-1}"
    assert 0 <= channel < C, f"channel {channel} out of range 0..{C-1}"

    T_plot = min(T, max_time_steps)

    if T_plot <= 0:
        raise ValueError(f"T_plot must be positive, got {T_plot}")

    real_seq = real_traj[seq_idx, :T_plot, channel]
    gen_seq = gen_traj[seq_idx, :T_plot, channel]

    fig, axes = plt.subplots(3, T_plot, figsize=(3 * T_plot, 9))

    if T_plot == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    mse_test_all = []

    for t_idx in range(T_plot):
        img_r = real_seq[t_idx]
        r_min, r_max = img_r.min(), img_r.max()
        img_r = (img_r - r_min) / (r_max - r_min + 1e-6)

        ax = axes[0, t_idx]
        ax.imshow(img_r.numpy(), cmap="viridis")
        ax.set_title(f"Real t={t_idx}")
        ax.axis("off")

        img_g = gen_seq[t_idx]
        g_min, g_max = img_g.min(), img_g.max()
        img_g = (img_g - g_min) / (g_max - g_min + 1e-6)

        ax = axes[1, t_idx]
        ax.imshow(img_g.numpy(), cmap="viridis")
        ax.set_title(f"Gen t={t_idx}")
        ax.axis("off")

        err = (real_seq[t_idx] - gen_seq[t_idx]) ** 2
        e_mean = err.mean()

        ax = axes[2, t_idx]
        ax.imshow(err.numpy(), cmap="viridis")
        ax.set_title(f"SE t={t_idx}, MSE(x1e-3)={e_mean.item() * 1e3:.4f}")
        ax.axis("off")

        mse_test_all.append(e_mean.item())

    plt.tight_layout()

    if saveloc is not None:
        os.makedirs(saveloc, exist_ok=True)
        save_path = os.path.join(
            saveloc,
            f"rollouts_traj{seq_idx}_ch{channel}_{extra_note}.png",
        )
        plt.savefig(save_path, dpi=150)

    plt.close()

    return mse_test_all