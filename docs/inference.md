# Inference, evaluation, and CARMA

`infer` and `evaluate` both invoke the selected implementation's original inference entry point. `evaluate` is a naming convenience; it does not replace source metrics or add a cross-model evaluator.

## Uni-DiT

```bash
python scripts/run.py infer dit-unidirectional -- \
  --ckpt /path/to/uni-dit.pth \
  --dataset_root /path/to/data-root \
  --samp_diff_steps 100 --ddim_eta 0
```

The supplied `infer_dit.py` calls a shared `run_all` routine that includes backward prediction and cyclic rollout, even though the designated Uni-DiT trainer exposes only forward pairs with `[1,0]`. This is a pre-existing scientific mismatch. It remains unchanged for source integrity, but its backward/cyclic outputs must not be reported as valid Uni-DiT CARMA results.

For forward-only paper evaluation, use the snapshot's dedicated reproduction script:

```bash
python scripts/run.py reproduce dit-unidirectional \
  scripts/evaluate_unidirectional_dit_multirun.py -- \
  --ckpt /path/to/uni-dit.pth --dataset_root /path/to/data-root
```

## Bi-DiT

```bash
python scripts/run.py infer dit-bidirectional -- \
  --ckpt /path/to/bi-dit.pth \
  --dataset_root /path/to/data-root \
  --samp_diff_steps 1000 --cyclic_runs 4 8 16
```

The source evaluator runs forward/backward next-step tests, autoregressive rollouts, and fixed-budget cyclic candidates using the direction-conditioned checkpoint.

The equivalent explicit CARMA wrapper is:

```bash
python scripts/run.py carma dit-bidirectional -- \
  --ckpt /path/to/bi-dit.pth --dataset_root /path/to/data-root --cyclic_runs 4 8 16
```

## Uni-FlowFormer

```bash
python scripts/run.py infer flowformer-unidirectional -- \
  --ckpt /path/to/flowformer-forward.pth --num_ode_steps 5

python scripts/run.py infer flowformer-unidirectional -- \
  --ckpt /path/to/flowformer-backward.pth --num_ode_steps 5
```

The direction is read from checkpoint metadata. Inference prefers `vids_test_<dataset>_<direction>.npy` and warns before falling back to the legacy non-direction-specific filename. The source contains next-step and rollout evaluation for one checkpoint direction at a time; it does not contain a supported two-checkpoint CARMA selector. The common launcher therefore rejects `carma flowformer-unidirectional`.

## Bi-FlowFormer

```bash
python scripts/run.py infer flowformer-bidirectional -- \
  --ckpt /path/to/bi-flowformer.pth \
  --num_ode_steps 5 --cyclic_runs 4 8 16

python scripts/run.py carma flowformer-bidirectional -- \
  --ckpt /path/to/bi-flowformer.pth \
  --num_ode_steps 5 --cyclic_runs 4 8 16
```

Both commands invoke the original `infer_flowformer.py`. The model uses `[1,0]` for forward and `[0,1]` for backward integration.

## Checkpoint behavior

No pretrained checkpoint is currently available for download. A Hugging Face release is planned after the publication.

The original inference classes expect a checkpoint containing `args` and `model_state_dict`. They construct the architecture from saved dataset/model-size metadata, strip a leading DataParallel `module.` prefix when present, and load weights strictly. Preserve at least these metadata fields:

- dataset and model-size identifiers;
- patch size and time-embedding dimension;
- DiT architecture/diffusion configuration where applicable;
- Uni-FlowFormer direction.

Use absolute checkpoint paths to avoid implementation-specific fallback directories. No original checkpoint was supplied, so historical checkpoint loading is **unverified**.

## Sampling controls

- `samp_diff_steps` controls DiT reverse diffusion/DDIM steps.
- `ddim_eta` controls DiT stochasticity.
- `num_ode_steps` controls FlowFormer explicit-Euler integration.

An ODE step count and a diffusion reverse-step count are not interchangeable.

## Cyclic scoring semantics

The bidirectional compatibility evaluators compute:

- endpoint error: MSE between the reconstructed final backward state and the initial forward state;
- path-wise error: MSE between the backward trajectory and the time-reversed forward trajectory.

These cyclic scores do not use future ground truth for test-time ranking. The compatibility `cyclic_rollout_multi` implementation averages each candidate's error over evaluated trajectories and selects one global candidate. It also regenerates an independent candidate pool for each requested `K`. Paper tensor scripts use a different source-backed protocol: one shared `K_max` pool and per-trajectory ranking. Keep these protocols distinct.

Ground-truth oracle scripts under the bidirectional FlowFormer snapshot are evaluation-only baselines and must never be described as deployable CARMA selection.

## Results

Result tensors, plots, CSV files, and normalized test arrays are written wherever the original implementation specifies—normally its local `results/` or `datasets/` directory. The external wrapper does not rewrite these paths. The source inference modules and argument parsers passed import/`--help` checks in a temporary dependency environment, but no real inference run was completed because trained weights and datasets were absent.

