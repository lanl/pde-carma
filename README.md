# pde-carma

`pde-carma` brings four CARMA PDE-forecasting implementations into one
repository.

The original code came from two model repositories—DiT and FlowFormer
(FMT)—with unidirectional and bidirectional branches in each. The four source
snapshots are kept separately under `implementations/`. Their model code,
training logic, data loaders, samplers, and checkpoint formats are unchanged.

## Included models

| Launcher name | What it trains |
|---|---|
| `dit-unidirectional` | A forward-only DiT |
| `dit-bidirectional` | One direction-conditioned DiT |
| `flowformer-unidirectional` | Separate forward and backward FlowFormer models |
| `flowformer-bidirectional` | One direction-conditioned FlowFormer |

CARMA cyclic selection is available only for the bidirectional models.

## Setup

Each implementation contains its original `environment.yml`. Create separate
DiT and FlowFormer environments when exact experiment isolation matters, then
install a PyTorch build suitable for your machine. PyTorch is not included in
the supplied environment files.

The common launcher can run directly from the checkout. Installing it is
optional:

```bash
python -m pip install -e . --no-deps
python scripts/run.py list
```

Activate the model environment before training, or pass
`--python /path/to/environment/python` before the `train` or `infer` command.
See [installation.md](docs/installation.md) for details.

## Data

Datasets and dataset download scripts are not included. Store shared data under
`datasets/`; the exact filenames and supported datasets are listed in
[datasets.md](docs/datasets.md).

| Workflow | Dataset location |
|---|---|
| Both DiT variants | `/path/to/pde-carma/datasets/`; pass `--dataset_root /path/to/pde-carma` |
| Uni-FlowFormer training | `/path/to/pde-carma/datasets/`; pass `--dataset_root /path/to/pde-carma` |
| Uni-FlowFormer inference | `implementations/flowformer_unidirectional/datasets/` |
| Bi-FlowFormer training and inference | `implementations/flowformer_bidirectional/datasets/` |

For an implementation-local path, temporarily copy or link the required data
directory. Remove that copy or link before running strict source-integrity
checks because it adds files below `implementations/`.

## Training

Arguments after `--` are passed unchanged to the original training script.

```bash
# DiT
python scripts/run.py train dit-unidirectional -- --ds ce-rm --dataset_root /path/to/pde-carma
python scripts/run.py train dit-bidirectional -- --ds ce-rm --dataset_root /path/to/pde-carma

# Uni-FlowFormer needs one run per direction
python scripts/run.py train flowformer-unidirectional -- --direction forward --ds ce-rm --dataset_root /path/to/pde-carma
python scripts/run.py train flowformer-unidirectional -- --direction backward --ds ce-rm --dataset_root /path/to/pde-carma

# Bidirectional FlowFormer
python scripts/run.py train flowformer-bidirectional -- --ds ce-rm
```

The original trainers save checkpoints under their implementation's `models/`
directory, sometimes inside a run-specific subdirectory. More options are in
[training.md](docs/training.md).

## Checkpoints and inference

You can train checkpoints locally. Pretrained checkpoints are planned for
Hugging Face, but they are **not available yet**. There is no verified
repository ID, filename list, or checksum at this time.

After the Hugging Face repository is published, the download will follow this
pattern:

```bash
python -m pip install huggingface_hub
hf download ORG/REPO --local-dir artifacts/checkpoints
```

`ORG/REPO` is a placeholder and is not currently usable. Store downloaded
weights under `artifacts/checkpoints/` and pass an absolute path to `--ckpt`.
Uni-FlowFormer requires separate forward and backward checkpoints. Do not mix
checkpoints between variants.

```bash
# DiT example
python scripts/run.py infer dit-bidirectional -- \
  --ckpt /absolute/path/to/model.pth \
  --dataset_root /path/to/pde-carma

# FlowFormer example
python scripts/run.py infer flowformer-bidirectional -- \
  --ckpt /absolute/path/to/model.pth
```

The unchanged Uni-DiT generic inference script also runs backward and cyclic
routines even though its trainer is forward-only. Use the forward-only
evaluation workflow in [inference.md](docs/inference.md) for Uni-DiT results.

## CARMA

```bash
python scripts/run.py carma dit-bidirectional -- \
  --ckpt /absolute/path/to/model.pth \
  --dataset_root /path/to/pde-carma \
  --cyclic_runs 4 8 16

python scripts/run.py carma flowformer-bidirectional -- \
  --ckpt /absolute/path/to/model.pth \
  --cyclic_runs 4 8 16
```

## Quick checks

Check command resolution without starting a run:

```bash
python scripts/run.py --dry-run train dit-unidirectional -- --ds ce-rm
```

The model smoke test requires PyTorch. The data-loader smoke test requires
PyTorch, NumPy, and xarray.

```bash
python tools/smoke_models.py
python tools/smoke_data_loaders.py
python -m unittest discover -s tests
```

## Documentation

- [Installation](docs/installation.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Inference and evaluation](docs/inference.md)
- [Repository structure](docs/directory_structure.md)

## Release status

**Research preview: code available, pretrained assets pending.**

Datasets and pretrained checkpoints are not included. Full training,
historical checkpoint loading, numerical parity, and paper-result reproduction
have not yet been verified in this combined repository.

## License

EIDR O#5133: *conditional-flow-matching on open-pde-datasets*. This program is
released under the MIT License.
