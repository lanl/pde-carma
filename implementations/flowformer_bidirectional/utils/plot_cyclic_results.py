#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== CONFIG ======
folder = Path(r"F:\FM\codes\lisdi\pde-reasoning\experiments\results_flowmatching_video")  
basech = 64
T = 20  # number of values per file
# ====================

def load_vec16(txt_path: Path, T=20):
    """Load a txt file"""
    arr = np.loadtxt(txt_path)
    arr = np.asarray(arr).reshape(-1)  # flatten handles (16,), (1,16), etc.
    return arr

def load_group(pattern: str):
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern} in {folder}")
    data = np.stack([load_vec16(f, T) for f in files], axis=0)  # (N_runs, 16)
    return files, data

# --- 1) rollout dir=1 (16 runs) ---
files_p, data_p = load_group(f"mse_rollout_dir1_basech{basech}_run*.txt")
print(f"Data shape for dir=1: {data_p.shape}")
best_p = data_p.min(axis=0)
mean_p = data_p.mean(axis=0)
std_p  = data_p.std(axis=0, ddof=1) if data_p.shape[0] > 1 else np.zeros(T)

# get the best forward trajectory across runs
run_score = data_p.mean(axis=1)          # (16,)
best_idx = int(np.argmin(run_score))
best_curve_p = data_p[best_idx]            # (20,)
print("[forward] best run idx:", best_idx, "score:", run_score[best_idx])

# --- 2) rollout dir=-1 (16 runs) ---
files_m, data_m = load_group(f"mse_rollout_dir-1_basech{basech}_run*.txt")
print(f"Data shape for dir=-1: {data_m.shape}")
mean_m = data_m.mean(axis=0)
std_m  = data_m.std(axis=0, ddof=1) if data_m.shape[0] > 1 else np.zeros(T)

# get the best backward trajectory across runs
run_score = data_m.mean(axis=1)          # (16,)
best_idx = int(np.argmin(run_score))
best_curve_m = data_m[best_idx]            # (20,)
print("[backward] best run idx:", best_idx, "score:", run_score[best_idx])

# --- 3) cyclic best (single file) ---
cyc_files = sorted(folder.glob("mse_cyclic_best_wrt_gt*.txt"))
if not cyc_files:
    raise FileNotFoundError(f"No files match: mse_cyclic_best_wrt_gt*.txt in {folder}")
cyc = load_vec16(cyc_files[0], T)[:-1]  # skip last value
print(f"Loaded cyclic best file: {cyc_files[0].name} with shape {cyc.shape}")

# x-axis: 20 values
x = np.arange(1, T + 1)

plt.figure(figsize=(12, 6))

# forward mean + band
plt.plot(x, 1000 * mean_p, label="Rollout dir=1 (mean)")
plt.fill_between(x, 1000*(mean_p-std_p), 1000*(mean_p+std_p), alpha=0.15)

# # backward mean + band
# plt.plot(x, 1000 * mean_m, label="Rollout dir=-1 (mean)")
# plt.fill_between(x, 1000*(mean_m-std_m), 1000*(mean_m+std_m), alpha=0.15)

# cyclic best
plt.plot(x, 1000 * cyc, marker="o", linestyle="-", label="Cyclic best")

# best rollouts
# plt.plot(x, 1000 * best_curve_p, marker="x", linestyle="-", label="Rollout dir=1 (best)")
# plt.plot(x, 1000 * best_curve_m, marker="x", linestyle="-", label="Rollout dir=-1 (best)")

plt.xlabel("Time steps", fontsize=18)
plt.ylabel("MSE × 1000", fontsize=18)
plt.title("MSE across rollout horizon (single test set example)", fontsize=20)

plt.xticks(np.arange(1, T + 1, 1), fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, alpha=0.25)

ax = plt.gca()
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False, fontsize=16)

plt.subplots_adjust(bottom=0.25)  # make room for legend
plt.tight_layout()

out_path = folder / "mse_summary_plot.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()


# %% box plots
# %% box plots
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ========= CONFIG =========
folder = Path(r"F:\FM\codes\lisdi\pde-reasoning\experiments\results_flowmatching_video")
basech = 64

pattern_fwd = f"mse_rollout_dir1_basech{basech}_run*.txt"
pattern_bwd = f"mse_rollout_dir-1_basech{basech}_run*.txt"

cyclic_file = "mse_cyclic_top8_wrt_gt.txt"  # expects (K,T) or (K,T+1) or (T,) / (T+1,)
topk = 3                                    # plot top-3 cyclic curves

scale = 1000.0  # set to 1.0 if you don't want scaling
# ==========================

def load_curve(path: Path) -> np.ndarray:
    """Load a txt file containing (T,) or (N,T). If (N,T), average over N -> (T,)."""
    arr = np.loadtxt(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.reshape(-1)

def load_runs(pattern: str):
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} in {folder}")

    curves = [load_curve(f) for f in files]
    T = curves[0].size
    if any(c.size != T for c in curves):
        bad = [(files[i].name, curves[i].size) for i in range(len(curves)) if curves[i].size != T]
        raise ValueError(f"Inconsistent T across runs for pattern {pattern}: {bad}")

    data = np.stack(curves, axis=0)  # (runs, T)
    return files, data

def boxplot_per_timestep(ax, data_runs_T, title):
    """One box per timestep; stats computed over runs. Shows both median and mean."""
    runs, T = data_runs_T.shape
    x = np.arange(1, T + 1)

    data_list = [scale * data_runs_T[:, t] for t in range(T)]  # list of length T

    bp = ax.boxplot(
        data_list,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        showmeans=True,          # <-- show mean
        meanline=True,           # <-- mean as a line
        medianprops=dict(color="C1", linewidth=2),              # median line
        meanprops=dict(color="k", linestyle="--", linewidth=2), # mean line
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # style boxes (make them consistent)
    for b in bp["boxes"]:
        b.set(facecolor="C0", alpha=0.30, edgecolor="C0")

    ax.set_title(title)
    ax.set_ylabel(f"MSE × {int(scale)}" if scale != 1 else "MSE")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(0.4, T + 0.6)

    return x

# ---- Load data ----
files_f, data_f = load_runs(pattern_fwd)   # (runs, T)
files_b, data_b = load_runs(pattern_bwd)   # (runs, T)

# ---- Load cyclic top-K curves ----
cyc_path = folder / cyclic_file
if not cyc_path.exists():
    raise FileNotFoundError(f"Cyclic file not found: {cyc_path}")

# ---- Align lengths ----
T = data_f.shape[1]
if data_b.shape[1] != T:
    raise ValueError(f"Forward T={T} but backward T={data_b.shape[1]}")

cyc = np.loadtxt(cyc_path)
cyc = np.asarray(cyc)
if cyc.ndim == 1:
    cyc = cyc[None, :]  # (1, T) or (1, T+1)

# handle (K, T+1) -> drop last column
if cyc.shape[1] == T + 1:
    cyc = cyc[:, :-1]

if cyc.shape[1] != T:
    raise ValueError(f"Cyclic shape={cyc.shape}, expected (*, {T}) (or (*, {T+1}))")

cyc_top = cyc[:min(topk, cyc.shape[0])]  # (<=3, T)

x = np.arange(1, T + 1)

# ---- Plot: two subplots, cyclic overlay on both ----
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

boxplot_per_timestep(axs[0], data_f, f"Forward model: per-time-step MSE over {data_f.shape[0]} runs")

cyc_lines0 = []
for i in range(cyc_top.shape[0]):
    line, = axs[0].plot(
        x, scale * cyc_top[i],
        marker="o", linewidth=2,
        label=f"Cyclic (top {i+1})",
        zorder=5
    )
    cyc_lines0.append(line)

boxplot_per_timestep(axs[1], data_b, f"Backward model: per-time-step MSE over {data_b.shape[0]} runs")

cyc_lines1 = []
for i in range(cyc_top.shape[0]):
    line, = axs[1].plot(
        x, scale * cyc_top[i],
        marker="o", linewidth=2,
        label=f"Cyclic (top {i+1})",
        zorder=5
    )
    cyc_lines1.append(line)

axs[1].set_xlabel("Time step")

# ---- Legends (include box, median, mean, cyclic top-k) ----
legend_base = [
    Patch(facecolor="C0", edgecolor="C0", alpha=0.30, label="Box (IQR over runs)"),
    Line2D([0], [0], color="C1", linewidth=2, label="Median"),
    Line2D([0], [0], color="k", linestyle="--", linewidth=2, label="Mean"),
]
axs[0].legend(handles=legend_base + cyc_lines0, loc="upper left")
axs[1].legend(handles=legend_base + cyc_lines1, loc="upper left")

plt.tight_layout()
out_path = folder / "mse_boxplots_forward_backward_with_cyclic_overlay_top3_mean_median.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", out_path)
print("Forward runs:", len(files_f), "Backward runs:", len(files_b), "Cyclic file:", cyc_path.name, "Plotted:", cyc_top.shape[0])

# %% plot mse and cyclic error
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

folder = Path(r"F:\FM\codes\lisdi\pde-reasoning\experiments\results_flowmatching_video")

cyclic_error_file = "cyclic_top64_idx_and_error.txt"
cyclic_mse_file   = "mse_cyclic_top64_wrt_gt.txt"

meta = np.loadtxt(folder / cyclic_error_file, skiprows=1) 
cyc_err = meta[:, 1]                                      # (64,)

mse = np.loadtxt(folder / cyclic_mse_file)
mse = np.atleast_2d(mse)                                  # (64,21)
print("MSE shape:", mse.shape)

for t in range(mse.shape[1]):
    mse_timestep = mse[:, t]
    mse_last = mse[:, -1]
    mse_avg = mse.mean(axis=1)                                # (64,)

    # select what MSE to use
    mse_plot = mse_timestep

    # pearson and spearman correlation
    pear_r, pear_p = pearsonr(cyc_err, mse_plot)
    spear_rho, spear_p = spearmanr(cyc_err, mse_plot)

    # figure
    plt.figure(figsize=(6,5))
    plt.scatter(cyc_err, mse_plot, s=35)
    plt.xlabel("Cyclic error")
    plt.ylabel("Mean MSE over time")
    plt.title(f"Cyclic error vs mean MSE | Pearson r={pear_r:.3f}, Spearman ρ={spear_rho:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
