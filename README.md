# pde-carma

This repository brings the four PDE forecasting implementations used by CARMA into one checkout. The practical goal is simple: give each model a consistent way to launch while leaving the research code exactly as it was supplied.

The upstream code was organized as two model-family repositories: DiT and
FlowFormer (FMT). Each repository had separate unidirectional and bidirectional
branches. This project combines snapshots of those four branches into a single
repository while keeping every implementation independent and unchanged.

Each source snapshot is copied byte-for-byte under `implementations/`. The launchers, tests, integrity tools, and documentation live outside those directories. We did not rewrite or deduplicate any model architecture, training loop, sampler, evaluator, data loader, default, or checkpoint format.

## Implementations

| Wrapper name | Original training entry point | Original inference entry point | Preserved semantics |
|---|---|---|---|
| `dit-unidirectional` | `train_dit_unidirectional.py` | `infer_dit.py` | Forward-only pairs; fixed control `[1,0]` |
| `dit-bidirectional` | `train_dit.py` | `infer_dit.py` | One direction-conditioned diffusion model |
| `flowformer-unidirectional` | `train_flowformer.py --direction forward\|backward` | `infer_flowformer.py` | Separate forward and backward checkpoints |
| `flowformer-bidirectional` | `train_flowformer.py` | `infer_flowformer.py` | One direction-conditioned flow model |

The unidirectional implementations are not silently converted into bidirectional models. CARMA selection is exposed only for the two source implementations that contain a bidirectional cyclic evaluation path.

## Quick start

The external launcher uses only the Python standard library. It changes into the selected implementation directory, invokes its original entry point with the selected Python executable, and forwards source arguments unchanged.

```bash
# Show registered variants and source entry points.
python scripts/run.py list

# Equivalent after an editable install (or with PYTHONPATH=src):
python -m pde_carma list

# Inspect resolution and argument forwarding without executing source code.
python scripts/run.py --dry-run train dit-unidirectional -- --ds ce-rm --epochs 2000

# Train the four modes. Uni-FlowFormer is run once per direction.
python scripts/run.py train dit-unidirectional -- --ds ce-rm --dataset_root /path/to/data-root
python scripts/run.py train dit-bidirectional -- --ds ce-rm --dataset_root /path/to/data-root
python scripts/run.py train flowformer-unidirectional -- --direction forward --ds ce-rm --dataset_root /path/to/data-root
python scripts/run.py train flowformer-unidirectional -- --direction backward --ds ce-rm --dataset_root /path/to/data-root
python scripts/run.py train flowformer-bidirectional -- --ds ce-rm

# Run the unchanged inference entry points.
python scripts/run.py infer dit-bidirectional -- --ckpt /path/to/checkpoint.pth --dataset_root /path/to/data-root --cyclic_runs 4 8 16
python scripts/run.py infer flowformer-bidirectional -- --ckpt /path/to/checkpoint.pth --cyclic_runs 4 8 16
```

Use `--python /path/to/python` before the subcommand to select a model environment. Everything after the separator `--` is passed to the original script. The eight convenience launchers under `scripts/train/` and `scripts/infer/` provide the same preservation-oriented behavior.

## Repository layout

```text
pde-carma/
├── implementations/              # Immutable source snapshots
│   ├── dit_unidirectional/
│   ├── dit_bidirectional/
│   ├── flowformer_unidirectional/
│   └── flowformer_bidirectional/
├── src/pde_carma/                # External registry and subprocess launcher
├── scripts/                      # Unified and variant-specific wrappers
├── tools/                        # Source-integrity verification
├── tests/                        # Wrapper/integrity tests; no source edits
└── docs/                         # Installation, workflows, reports
```

See [the documented structure](docs/directory_structure.md) for responsibilities and the exact source-root mapping.

## Installation

The four snapshots retain their own environment files and do not have a verified universal environment. Their runtime imports include PyTorch, NumPy, xarray, and Matplotlib; reproduction utilities may require SciPy, pandas, and other packages. Install a platform-appropriate PyTorch build separately rather than assuming a CUDA version.

```bash
# Optional: install the external launcher itself.
python -m pip install -e . --no-deps
```

The implementation environment files do not include every runtime dependency, notably PyTorch. See [installation.md](docs/installation.md) before creating an experiment environment.

## Where to put the datasets

Dataset files are not included, and the supplied snapshots do not contain a
verified download script or complete upstream URL list. The layout below is
therefore about placement, not acquisition. A convenient shared location is
the repository's top-level `datasets/` directory:

```text
pde-carma/
`-- datasets/
    |-- pdegym/
    |   `-- CE-RM/CE-RM.nc
    |-- pdebench/
    |   |-- vids_train_2d-reacdiff.npy
    |   |-- vids_val_2d-reacdiff.npy
    |   `-- vids_test_2d-reacdiff.npy
    `-- thewell/
        |-- vids_train_active-matter.npy
        |-- vids_val_active-matter.npy
        `-- vids_test_active-matter.npy
```

The original programs do not all resolve data paths in the same way:

| Workflow | Dataset location and command-line behavior |
|---|---|
| Uni-DiT train/infer | Keep data in `<repo>/datasets` and pass `--dataset_root <repo>` |
| Bi-DiT train/infer | Keep data in `<repo>/datasets` and pass `--dataset_root <repo>` |
| Uni-FlowFormer training | Keep data in `<repo>/datasets` and pass `--dataset_root <repo>` |
| Uni-FlowFormer inference | The unchanged script reads `implementations/flowformer_unidirectional/datasets/`; it has no `--dataset_root` option |
| Bi-FlowFormer train/infer | The unchanged scripts read `implementations/flowformer_bidirectional/datasets/`; they have no `--dataset_root` option |

For the two implementation-local cases, copy the required files there or create a temporary directory link to the shared top-level data. Those local data directories are ignored by Git, but they are extra files relative to the immutable source snapshot; remove or unmount them before running the strict source-integrity check. The complete filename list, supported datasets, channels, and configured splits are in [datasets.md](docs/datasets.md).

## Checkpoints

Pretrained checkpoints are not bundled. They are intended to be distributed through Hugging Face, but that download is **not available yet**: there is currently no verified repository ID, URL, filename list, or published checksum to put in this README.

Once the Hugging Face repository is published, the expected workflow will use
the documented [Hugging Face `hf download` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m pip install -U huggingface_hub
hf download <org-or-user>/<repository> --local-dir artifacts/checkpoints
```

`<org-or-user>/<repository>` is deliberately a placeholder; do not treat it as a working address. Checkpoints may be arranged however is convenient under `artifacts/checkpoints/`, for example:

```text
artifacts/checkpoints/
|-- dit_unidirectional/
|-- dit_bidirectional/
|-- flowformer_unidirectional/
|   |-- forward/
|   `-- backward/
`-- flowformer_bidirectional/
```

The original scripts do not require those folder names. Pass the downloaded file explicitly with an absolute `--ckpt` path. Uni-FlowFormer needs separate forward and backward checkpoints; each of the other variants uses its own checkpoint and formats must not be mixed. When files become available, the release instructions should also publish SHA-256 checksums and identify the dataset, architecture, direction, and sampling configuration for every checkpoint. See [inference.md](docs/inference.md) for the metadata expected inside each file.

## Reproduction and validation

- [Training](docs/training.md)
- [Inference and evaluation](docs/inference.md)
- [Paper settings and reproduction scripts](docs/reproducibility.md)
- [Installation details](docs/installation.md)
- [Source-integrity report](docs/source_integrity.md)
- [Validation report](docs/validation_report.md)
- [Publication-readiness blockers](docs/publication_readiness.md)

The integrity checker compares every copied file against a source-derived SHA-256 manifest:

```bash
python tools/verify_source_integrity.py
```

With PyTorch installed, a lightweight CPU smoke test can exercise all four
preserved model modules without a dataset or checkpoint:

```bash
python tools/smoke_models.py

# Or point it at a separate model environment.
python tools/smoke_models.py --python /path/to/model-env/bin/python --json
```

Each variant runs in its own subprocess. The test constructs a tiny model,
runs a synthetic forward and backward pass, and takes one native DDIM or ODE
sampling step. It is useful for catching import, tensor-shape, and basic runtime
errors; it is not a substitute for loading the real data and checkpoints.

A second smoke test covers the preserved NPY loaders and training-pair logic.
It requires PyTorch, NumPy, and xarray, but still needs no real dataset:

```bash
python tools/smoke_data_loaders.py

# Or use a separate model environment.
python tools/smoke_data_loaders.py --python /path/to/model-env/bin/python --json
```

It creates small arrays in a temporary directory, checks train-statistic
normalization and batch shapes, verifies forward-only Uni-DiT pairs,
forward/backward Bi-DiT and Bi-FlowFormer controls, and Uni-FlowFormer's
direction-specific time reversal. It never writes into `implementations/`.

No datasets, trained checkpoints, or experiment outputs were supplied, so full training, historical checkpoint loading, numerical parity, and paper-result reproduction are not claimed.

## License, citation, and release status

Note: EIDR number O#5133 - conditional-flow-matching on open-pde-datasets. This program is Open-Source under the MIT License.

