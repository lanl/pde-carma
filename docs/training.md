# Training

The external launcher calls each original training file in its own directory and forwards arguments verbatim. Put wrapper options before the subcommand and source options after `--`.

## Uni-DiT

```bash
python scripts/run.py train dit-unidirectional -- \
  --ds ce-rm --dataset_root /path/to/data-root --ms ti --patch_size 8 8
```

This resolves to `implementations/dit_unidirectional/train_dit_unidirectional.py`, not the generic `train_dit.py` that also exists in that snapshot. It constructs only `x_t -> x_{t+1}` pairs, retains the direction embedding, and supplies `[1,0]` for every training example.

Preserved source defaults:

- `num_diffusion_steps=1000`
- `samp_diff_steps=100`
- `beta_start=1e-4`, `beta_end=2e-2`, `ddim_eta=0`
- `epochs=2000`, batch size 256, learning rate `1e-4`

## Bi-DiT

```bash
python scripts/run.py train dit-bidirectional -- \
  --ds ce-rm --dataset_root /path/to/data-root --ms ti --patch_size 8 8
```

This resolves to `implementations/dit_bidirectional/train_dit.py`. Every adjacent pair appears in both directions with `[1,0]` and `[0,1]` controls.

Preserved source defaults:

- `num_diffusion_steps=100`
- `samp_diff_steps=1000`
- the same beta/DDIM, optimizer, epoch, and batch defaults as Uni-DiT

The apparently inverted Uni/Bi diffusion defaults are genuinely present in the snapshots. Do not normalize or swap them without author review. Both trainers expose `--dit_arch legacy|adaln_zero` and share checkpoint-compatible architecture code only within their own unchanged copies.

## Uni-FlowFormer

Train two independent checkpoints when both directions are required:

```bash
python scripts/run.py train flowformer-unidirectional -- \
  --direction forward --ds ce-rm --dataset_root /path/to/data-root --ms ti

python scripts/run.py train flowformer-unidirectional -- \
  --direction backward --ds ce-rm --dataset_root /path/to/data-root --ms ti
```

The architecture has no direction-control embedding. Backward training reverses trajectories before forming adjacent pairs. Checkpoint names and saved argument metadata include the selected direction. The preserved `num_ode_steps` default used for validation sampling is 5.

## Bi-FlowFormer

```bash
python scripts/run.py train flowformer-bidirectional -- \
  --ds ce-rm --ms ti --patch_size 8 8
```

This produces one direction-conditioned checkpoint. Flow matching preserves the source interpolation
`x_t=(1-t)x_0+t x_1`, target velocity `x_1-x_0`, MSE loss, and explicit-Euler validation sampler. Its preserved `num_ode_steps` default is 5.

## Devices, resume, and outputs

All source trainers retain `--device_idx`; `--parallel` uses `torch.nn.DataParallel` when multiple CUDA devices are available. Device ordering comes from the original `DeviceManager` and should be inspected on the target machine.

Resume uses the original pattern:

```bash
python scripts/run.py train MODEL -- --resume --ckpt /path/to/checkpoint.pth [other source flags]
```

Do not exchange checkpoints across variants. Bi-FlowFormer contains `control_mlp.*` parameters that Uni-FlowFormer does not; Uni-FlowFormer forward/backward checkpoints are separate. DiT checkpoint metadata also preserves each trainer's diffusion defaults and architecture selection.

The wrappers do not redirect output paths. Depending on the unchanged source entry point, checkpoints and plots are written beneath implementation-local `models/` and `results/` directories, either flat or nested by run name. These generated artifacts must remain untracked.

The main training entry points do not establish a global random seed. Record Python, NumPy, PyTorch, data-loader, and accelerator state externally for new experiments.

## Dependency-free command verification

Use dry-run mode to check entry-point resolution and exact argument order without importing PyTorch or touching data:

```bash
python scripts/run.py --dry-run train flowformer-unidirectional -- \
  --direction backward --ds fns-kf --num_ode_steps 100
```

