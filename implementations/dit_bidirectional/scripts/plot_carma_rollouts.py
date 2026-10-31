#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot test PDE rollouts: ground truth / evaluation-only Oracle / CARMA Cyclic."
    )
    parser.add_argument("--family", choices=["fmt", "dit"], help="Model family when generating new pools")
    parser.add_argument("--ckpt", help="Path to bidirectional model checkpoint")
    parser.add_argument("--dataset", help="Path to vids_test_<dataset>.npy (N,T,C,H,W)")
    parser.add_argument("--traj_indices", nargs="+", type=int, default=[52],
                        help="ZERO-based test trajectory indices")
    parser.add_argument("--K", type=int, default=8, help="Number of shared-pool forward/backward candidates")
    parser.add_argument("--max_T_rollout", type=int, default=20,
                        help="Number of frames INCLUDING observed t=0 (20 means t=0..19)")
    parser.add_argument("--num_ode_steps", type=int, default=100, help="FM Euler steps per physical-time step")
    parser.add_argument("--samp_diff_steps", type=int, default=100, help="DiT DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device_idx", type=int, default=0, help="GPU index; CPU if CUDA unavailable")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out_dir", default="results/paper_rollouts")
    parser.add_argument("--reuse_pt", nargs="+", type=str,
                        help="Plot one or more previously saved traj*_pool.pt files; skip model inference")
    parser.add_argument("--time_indices", nargs="+", type=int,
                        help="0-based physical frame numbers to DISPLAY, e.g. 0 4 8 12 16 19")
    parser.add_argument("--num_display_frames", type=int, default=6,
                        help="Evenly spaced display frames if --time_indices is omitted")
    parser.add_argument("--channel", type=int, default=0, help="Physical field channel to display")
    parser.add_argument("--rows", nargs="+", choices=["ground_truth", "forward", "oracle", "cyclic"],
                        default=["ground_truth", "oracle", "cyclic"],
                        help="Rows to show; 'forward' is first unselected candidate")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--color_scale", choices=["global", "per_frame"], default="global",
                        help="Shared GT-based color scale: all frames or separately per column, never per row")
    parser.add_argument("--percentiles", nargs=2, type=float, default=[0.5, 99.5],
                        metavar=("LOW", "HIGH"),
                        help="GT percentiles for color bounds (same bounds across rows)")
    parser.add_argument("--vmin", type=float, default=None, help="Optional fixed lower color bound")
    parser.add_argument("--vmax", type=float, default=None, help="Optional fixed upper color bound")
    parser.add_argument("--title", default=None, help="Optional figure header (e.g. CE-RM | BiFMT-3.5M)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution and PDF rasterized-image resolution (default: 300)")
    parser.add_argument("--colorbar_width", type=float, default=0.025,
                        help="Right-hand colorbar width as fraction of figure width (default: 0.025)")
    args = parser.parse_args(argv)
    if not args.reuse_pt and not all([args.family, args.ckpt, args.dataset]):
        parser.error("When not using --reuse_pt, supply --family, --ckpt, and --dataset")
    if args.K < 1 or args.max_T_rollout < 2:
        parser.error("--K must be >= 1 and --max_T_rollout must be >= 2")
    if args.num_ode_steps < 1 or args.samp_diff_steps < 1:
        parser.error("Sampling steps must be positive")
    if args.num_display_frames < 2:
        parser.error("--num_display_frames must be >= 2")
    lo, hi = args.percentiles
    if not (0 <= lo < hi <= 100):
        parser.error("--percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if (args.vmin is None) != (args.vmax is None):
        parser.error("Specify both --vmin and --vmax, or neither")
    if args.vmin is not None and args.vmin >= args.vmax:
        parser.error("--vmin must be smaller than --vmax")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if not (0.005 <= args.colorbar_width <= 0.05):
        parser.error("--colorbar_width must be between 0.005 and 0.05")
    return args


def resolve_existing_path(path):
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    p = ROOT / p
    if p.is_file():
        return p.resolve()
    raise FileNotFoundError(f"Could not find file: {path} (also tried {p})")


class _BidirectionalFMTViT(nn.Module):
    """Checkpoint-compatible ViTTime from the bidirectional FlowFormer branch.

    Defined here rather than importing ``utils.vit_flow.ViTTime`` because older
    local checkouts omit the direction embedding. Importing those checkouts
    causes the ``Unexpected key(s): control_mlp.*`` checkpoint loading error.

    The architecture and forward calculation here reproduce the uploaded
    pde-flowformer-bidirectional-flowformer/utils/vit_flow.py implementation.
    Other transformer and patch modules are still imported from the repo.
    """

    def __init__(self, in_channels, embed_dim, time_emb_dim, img_size,
                 patch_size, depth, num_heads):
        super().__init__()
        from utils.time_embedding import SinusoidalTimeEmbedding, TimeMLP
        from utils.pos_enc import LearnablePositionalEncoding
        from utils.transformer_main import TransformerBackbone
        from utils.patchify import PatchEmbed, PatchDecoder

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        else:
            patch_size = tuple(patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        else:
            img_size = tuple(img_size)

        self.physical_channels = in_channels
        self.in_channels = 2 * in_channels
        self.out_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.embed_dim = embed_dim

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = TimeMLP(time_emb_dim, time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, embed_dim)
        self.control_mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.patch_embed = PatchEmbed(
            in_channels=self.in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.pos_encoding = LearnablePositionalEncoding(
            grid_size=self.grid_size,
            embed_dim=embed_dim,
        )
        self.transformer = TransformerBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=0.0,
        )
        self.patch_decoder = PatchDecoder(
            out_channels=self.out_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

    def forward(self, x, t, c):
        # Same forward calculation as bidirectional utils/vit_flow.py.
        batch, channels, _, _ = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if c.ndim == 1:
            c = c.unsqueeze(0).expand(batch, -1)
        if c.shape != (batch, 2):
            raise ValueError(f"Expected direction control [{batch},2], got {tuple(c.shape)}")

        time_token = self.time_proj(self.time_mlp(self.time_embed(t)))
        control_token = self.control_mlp(c)
        tokens, hp, wp = self.patch_embed(x)
        tokens = self.pos_encoding(tokens, hp, wp)
        tokens = tokens + time_token[:, None, :] + control_token[:, None, :]
        tokens = self.transformer(tokens)
        return self.patch_decoder(tokens, hp, wp)


class _BidirectionalFMTSampler:
    """Minimal final-state sampler, retaining the original Euler update rule."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    @torch.inference_mode()
    def sample(self, x_cond, num_ode_steps=None, c=None):
        steps = num_ode_steps or self.config.get("num_ode_steps", 100)
        if steps < 1:
            raise ValueError("num_ode_steps must be positive")
        x_cond = x_cond.to(device=self.device, dtype=torch.float32)
        batch = x_cond.shape[0]
        if c is None:
            c = [1.0, 0.0]
        control = torch.as_tensor(c, dtype=x_cond.dtype, device=self.device)
        if control.ndim == 1:
            control = control.unsqueeze(0).expand(batch, -1)
        if control.shape != (batch, 2):
            raise ValueError(f"Invalid direction control shape {tuple(control.shape)}")

        x_flow = torch.randn_like(x_cond)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch,), i / steps,
                           device=self.device, dtype=x_cond.dtype)
            velocity = self.model(torch.cat([x_cond, x_flow], dim=1), t, control)
            x_flow = x_flow + dt * velocity
        # Returning the final state avoids copying every ODE substep to CPU.
        return x_flow


def _load_fmt_model(ckpt_path, shape_hw, device):
    """Load bidirectional weights strictly without relying on local vit_flow.py."""
    from utils.data_config import data_config
    from utils.model_config import model_config

    # The checkpoint is supplied by the user; keep loading behavior equivalent
    # to the repository's FlowFormerInference.from_pretrained.
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    config = checkpoint["args"]
    state = checkpoint["model_state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "control_mlp.0.weight" not in state:
        raise ValueError(
            "This checkpoint has no control_mlp weights. CARMA requires a "
            "bidirectional FlowFormer checkpoint (not a forward-only model)."
        )

    ds = config["ds"]
    scale = config["ms"]
    channels = data_config()[ds]["channels"]
    cfg = model_config()[scale]
    model = _BidirectionalFMTViT(
        in_channels=channels,
        embed_dim=cfg["model_dims"],
        time_emb_dim=config["time_emb_dim"],
        img_size=shape_hw,
        patch_size=config["patch_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )
    # STRICT loading is essential: don't silently discard direction weights.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Loaded bidirectional BiFMT checkpoint: dataset={ds}, model={scale}, "
          f"params={total_params / 1e6:.3f}M (strict=True).")
    return _BidirectionalFMTSampler(model, config, device)


def load_model(family, ckpt, shape_hw, device):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if family == "fmt":
        # Self-contained compatibility loader. Local model files stay unchanged.
        return _load_fmt_model(ckpt, shape_hw, device)
    from utils.my_model import DiTInference
    return DiTInference.from_pretrained(str(ckpt), img_size=shape_hw, device=device)


@torch.inference_mode()
def next_state(model, family, x_cond, direction, args):
    if family == "fmt":
        result = model.sample(x_cond=x_cond, num_ode_steps=args.num_ode_steps, c=direction)
        # Compatibility sampler returns only the final [B,C,H,W] state; the
        # repository's original sampler returns an ODE path [B,S,C,H,W].
        output = result[:, -1] if result.ndim == 5 else result
    else:
        # DiT can return the final state directly, avoiding 100 intermediate CPU copies.
        output = model.sample(x_cond=x_cond, samp_diff_steps=args.samp_diff_steps,
                              ddim_eta=args.ddim_eta, c=direction, return_path=False)
    if output.ndim != 4 or output.shape != x_cond.shape:
        raise RuntimeError(f"Unexpected sample shape: {tuple(output.shape)}, expected {tuple(x_cond.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Nonfinite model output: this checkpoint/trajectory may have diverged")
    return output.detach().cpu().float()


@torch.inference_mode()
def generate_pool(model, family, gt, args):
    """gt: [T,C,H,W]; return one record with all forward candidates [K,T,C,H,W]."""
    t_count = gt.shape[0]
    f_control = torch.tensor([1.0, 0.0], dtype=torch.float32)
    b_control = torch.tensor([0.0, 1.0], dtype=torch.float32)
    candidates = torch.empty((args.K, *gt.shape), dtype=torch.float32)
    scores_cyclic = torch.empty(args.K, dtype=torch.float32)
    scores_truth = torch.empty(args.K, dtype=torch.float32)

    for k in range(args.K):
        forward = torch.empty_like(gt)
        forward[0] = gt[0]
        cond = gt[0].unsqueeze(0)
        for t in range(1, t_count):
            cond = next_state(model, family, cond, f_control, args)
            forward[t] = cond[0]

        backward_rev = torch.empty_like(gt)
        backward_rev[0] = forward[-1]
        cond = forward[-1].unsqueeze(0)
        for s in range(1, t_count):
            cond = next_state(model, family, cond, b_control, args)
            backward_rev[s] = cond[0]

        # Measured cyclic score includes t=0 and t=T-1 (terminal contribution zero).
        scores_cyclic[k] = ((forward - backward_rev.flip(0)) ** 2).mean()
        # Evaluation-only error excludes known t=0, matching the paper tables.
        scores_truth[k] = ((forward[1:] - gt[1:]) ** 2).mean()
        candidates[k] = forward
        print(f"  candidate {k+1:2d}/{args.K}: cyclic={scores_cyclic[k].item():.7g} "
              f"true_MSE={scores_truth[k].item():.7g}", flush=True)

    oracle = int(torch.argmin(scores_truth).item())
    cyclic = int(torch.argmin(scores_cyclic).item())
    return {
        "ground_truth": gt.cpu(),
        "candidates": candidates.cpu(),
        "cyclic_scores": scores_cyclic.cpu(),
        "true_errors": scores_truth.cpu(),
        "oracle_index": oracle,
        "cyclic_index": cyclic,
        "forward_index": 0,
    }


def normalize_record(record):
    # Ensure no trust in saved indices: rederive using the same full-horizon metrics.
    gt = record["ground_truth"].detach().cpu().float()
    cand = record["candidates"].detach().cpu().float()
    if gt.ndim != 4 or cand.ndim != 5 or cand.shape[1:] != gt.shape:
        raise ValueError("Expected GT [T,C,H,W] and candidate pool [K,T,C,H,W]")
    true = (cand[:, 1:] - gt[None, 1:]).square().mean(dim=(1, 2, 3, 4))
    cyc = torch.as_tensor(record["cyclic_scores"]).flatten().float()
    if cyc.numel() != cand.shape[0] or not torch.isfinite(cyc).all():
        raise ValueError("Invalid cyclic scores")
    if not torch.isfinite(true).all() or not torch.isfinite(cand).all():
        raise ValueError("Nonfinite prediction or error in pool")
    record = dict(record)
    record["ground_truth"] = gt
    record["candidates"] = cand
    record["true_errors"] = true
    record["cyclic_scores"] = cyc
    record["oracle_index"] = int(torch.argmin(true))
    record["cyclic_index"] = int(torch.argmin(cyc))
    record["forward_index"] = 0
    return record


def selected_frames(t_count, args):
    if args.time_indices is not None:
        frames = list(args.time_indices)
    else:
        frames = np.linspace(0, t_count - 1, min(t_count, args.num_display_frames))
        frames = sorted(set(np.rint(frames).astype(int).tolist()))
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"--time_indices must be unique nonempty indices; got {frames}")
    if any(x < 0 or x >= t_count for x in frames):
        raise ValueError(f"Displayed time indices {frames} must lie in [0,{t_count-1}]")
    return frames


def to_python_floats(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return float(obj)
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def selection_gains(record):
    """Calculate per-trajectory gains against the SAME K-candidate Forward pool.

    Each E_k is the full-horizon MSE over t>=1, channels, and spatial pixels.
    E_forward = mean_k E_k, so gain = 100*(E_forward-E_selected)/E_forward.
    Oracle uses GT to evaluate the pool, never to choose the cyclic candidate.
    """
    errors = torch.as_tensor(record["true_errors"], dtype=torch.float64).flatten()
    if errors.numel() == 0 or not torch.isfinite(errors).all():
        raise ValueError("Expected finite true errors for all candidates")
    e_forward_mean = float(errors.mean().item())
    e_cyclic = float(errors[int(record["cyclic_index"])].item())
    e_oracle = float(errors[int(record["oracle_index"])].item())
    # If all candidates are exact, the percentage is 0 by convention (not 0/0).
    if e_forward_mean == 0.0:
        delta_cyclic = delta_oracle = 0.0
    else:
        delta_cyclic = 100.0 * (e_forward_mean - e_cyclic) / e_forward_mean
        delta_oracle = 100.0 * (e_forward_mean - e_oracle) / e_forward_mean
    return {
        "forward_mean_mse": e_forward_mean,
        "cyclic_mse": e_cyclic,
        "oracle_mse": e_oracle,
        "delta_cyc_percent": delta_cyclic,
        "delta_oracle_percent": delta_oracle,
    }


def plot_record(record, args, output_dir, index):
    record = normalize_record(record)
    gains = selection_gains(record)
    gt = record["ground_truth"]
    cand = record["candidates"]
    ti = selected_frames(gt.shape[0], args)
    if not (0 <= args.channel < gt.shape[1]):
        raise ValueError(f"Channel {args.channel} invalid for C={gt.shape[1]}")
    idx = {
        "forward": 0,
        "oracle": record["oracle_index"],
        "cyclic": record["cyclic_index"],
    }
    texts = {
        "ground_truth": "Ground truth",
        "forward": f"Forward (sample 1)\nMSE={record['true_errors'][0].item():.3g}",
        "oracle": f"Oracle (eval. only)\nMSE={record['true_errors'][idx['oracle']].item():.3g}",
        "cyclic": f"Cyclic (CARMA)\nMSE={record['true_errors'][idx['cyclic']].item():.3g}",
    }
    nrows = len(args.rows)
    ncols = len(ti)

    # Critical visual integrity: same color mapping for GT, Oracle, and Cyclic.
    gt_channel = gt[:, args.channel].numpy()
    if args.vmin is not None:
        lohi = [(args.vmin, args.vmax)] * ncols
    elif args.color_scale == "global":
        lo, hi = np.percentile(gt_channel, args.percentiles)
        lohi = [(float(lo), float(hi))] * ncols
    else:
        lohi = [tuple(float(x) for x in np.percentile(gt_channel[t], args.percentiles)) for t in ti]
    lohi = [(lo, hi if hi > lo else lo + 1e-6) for lo, hi in lohi]

    w = 2.0 * ncols + 2.25
    h = 2.12 * nrows + 1.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    fig.patch.set_facecolor("white")
    # Manual positioning remains consistent across narrow and wide configurations.
    fig.subplots_adjust(left=0.15, right=0.91, bottom=0.11, top=0.88,
                        wspace=0.035, hspace=0.065)

    for j, t in enumerate(ti):
        axes[0, j].set_title(f"$t={t}$", fontsize=13, pad=8)
    for i, row in enumerate(args.rows):
        if row == "ground_truth":
            seq = gt
        else:
            seq = cand[idx[row]]
        for j, t in enumerate(ti):
            axes[i, j].imshow(seq[t, args.channel].numpy(), origin="lower",
                              cmap=args.cmap, vmin=lohi[j][0], vmax=lohi[j][1],
                              interpolation="nearest", rasterized=True)
            axes[i, j].set_axis_off()

    # Add legible row labels at the center of each subplot row.
    fig.canvas.draw()
    for i, row in enumerate(args.rows):
        pos = axes[i, 0].get_position()
        fig.text(0.014, (pos.y0 + pos.y1) / 2, texts[row], fontsize=12,
                 ha="left", va="center")

    # For per-column normalization, show one colorbar only if it is global.
    if args.color_scale == "global" or args.vmin is not None:
        # Wider and taller than the previous 0.014 x 0.66 colorbar.
        ax_cb = fig.add_axes([0.93, 0.145, args.colorbar_width, 0.72])
        mappable = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(*lohi[0]),
                                         cmap=args.cmap)
        cb = fig.colorbar(mappable, cax=ax_cb)
        cb.set_label("Standardized field (channel {})".format(args.channel), fontsize=11)
        cb.ax.tick_params(labelsize=10)
    else:
        fig.text(0.94, 0.18, "GT-based\nscale per column", rotation=90,
                 fontsize=9, va="center", ha="center")

    label = args.title or record.get("label", "CARMA rollout")
    fig.suptitle(f"{label}  |  trajectory {index}  |  $K={cand.shape[0]}$",
                 fontsize=15, y=0.986)
    fig.text(0.53, 0.065,
             f"Cyclic gain: {gains['delta_cyc_percent']:+.2f}%    |    "
             f"Oracle gain: {gains['delta_oracle_percent']:+.2f}%  "
             "(vs. mean Forward MSE)",
             ha="center", va="center", fontsize=11, fontweight="semibold")
    fig.text(0.53, 0.032,
             "Selection uses all predicted frames; displayed snapshots are subsampled. "
             "Oracle is evaluation-only.",
             ha="center", va="center", fontsize=9)
    outstem = (
        f"traj{index:03d}_ch{args.channel}_K{cand.shape[0]}_"
        f"t{'-'.join(map(str, ti))}_{'-'.join(args.rows)}"
    )
    pdf_path = output_dir / f"{outstem}.pdf"
    png_path = output_dir / f"{outstem}.png"
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)

    summary = {
        "trajectory_index": index,
        "K": int(cand.shape[0]),
        "all_frames": int(gt.shape[0]),
        "displayed_frames": ti,
        "channel": args.channel,
        "oracle_candidate_index_zero_based": idx["oracle"],
        "cyclic_candidate_index_zero_based": idx["cyclic"],
        "forward_candidate_index_zero_based": 0,
        "forward_mean_mse": gains["forward_mean_mse"],
        "forward_mse": float(record["true_errors"][0]),
        "oracle_mse": gains["oracle_mse"],
        "cyclic_mse": gains["cyclic_mse"],
        "delta_cyc_percent": gains["delta_cyc_percent"],
        "delta_oracle_percent": gains["delta_oracle_percent"],
        "gain_baseline": "Mean full-horizon MSE of K forward candidates (predicted frames t>=1)",
        "output_dpi": args.dpi,
        "all_true_errors": record["true_errors"].tolist(),
        "all_cyclic_scores": record["cyclic_scores"].tolist(),
        "source_checkpoint": record.get("checkpoint"),
        "source_dataset": record.get("dataset"),
        "seed": record.get("seed"),
    }
    json_path = output_dir / f"{outstem}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=to_python_floats)
    print("  saved:", png_path)
    print("  saved:", pdf_path)
    print("  saved:", json_path)
    print(f"  oracle candidate={idx['oracle'] + 1}, MSE={summary['oracle_mse']:.7g}; "
          f"cyclic candidate={idx['cyclic'] + 1}, MSE={summary['cyclic_mse']:.7g}")
    print(f"  mean Forward MSE={summary['forward_mean_mse']:.7g}; "
          f"Cyclic gain={summary['delta_cyc_percent']:+.2f}%; "
          f"Oracle gain={summary['delta_oracle_percent']:+.2f}%")
    return summary


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_pt:
        for saved in args.reuse_pt:
            p = resolve_existing_path(saved)
            record = torch.load(str(p), map_location="cpu", weights_only=False)
            index = int(record.get("trajectory_index", -1))
            print(f"Re-plotting trajectory {index} from {p}")
            plot_record(record, args, out_dir, index)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f"cuda:{args.device_idx}")
    else:
        print("WARNING: No CUDA device detected; model inference will run on CPU.")
        device = torch.device("cpu")

    ckpt = resolve_existing_path(args.ckpt)
    dset = resolve_existing_path(args.dataset)
    data = np.load(str(dset), mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected N,T,C,H,W test data, got {data.shape}")
    n_total, t_total, channels, height, width = data.shape
    if args.channel < 0 or args.channel >= channels:
        raise ValueError(f"Channel {args.channel} invalid for dataset C={channels}")
    for index in args.traj_indices:
        if not (0 <= index < n_total):
            raise ValueError(f"Requested trajectory {index} not in [0,{n_total-1}]")
    nframes = min(args.max_T_rollout, t_total)
    selected_frames(nframes, args)  # fail fast before costly inference
    print(f"Model family={args.family}; checkpoint={ckpt}; dataset={dset}")
    print(f"Test set shape={data.shape}; scoring frames=0..{nframes-1}; K={args.K}")
    print(f"Device={device}; seed={args.seed}; selected trajectory indices={args.traj_indices}")
    model = load_model(args.family, ckpt, (height, width), device)

    with torch.inference_mode():
        for index in args.traj_indices:
            gt = torch.from_numpy(np.array(data[index, :nframes], copy=True)).float()
            print(f"\nGenerating trajectory {index}...", flush=True)
            record = generate_pool(model, args.family, gt, args)
            record.update({
                "trajectory_index": index,
                "family": args.family,
                "dataset": str(dset),
                "checkpoint": str(ckpt),
                "seed": args.seed,
                "num_ode_steps": args.num_ode_steps if args.family == "fmt" else None,
                "samp_diff_steps": args.samp_diff_steps if args.family == "dit" else None,
                "ddim_eta": args.ddim_eta if args.family == "dit" else None,
                "label": args.title or f"{args.family.upper()} / {dset.stem}",
            })
            pool_path = out_dir / f"traj{index:03d}_K{args.K}_pool.pt"
            torch.save(record, pool_path)
            print("  saved reusable full candidate pool:", pool_path)
            plot_record(record, args, out_dir, index)


if __name__ == "__main__":
    main()
