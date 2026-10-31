# Installation

There are two layers: the dependency-light external launcher and the model environments retained in each source snapshot.

## External launcher

The launcher uses the Python standard library and can run directly from a checkout:

```bash
python scripts/run.py list
python scripts/run.py --dry-run train dit-unidirectional -- --help
```

An editable installation exposes the same package without installing model dependencies:

```bash
python -m pip install -e . --no-deps
```

Use `--python` to run an original implementation with a different interpreter:

```bash
python scripts/run.py --python /path/to/model-env/bin/python train dit-bidirectional -- --help
```

On Windows, pass the full path to that environment's `python.exe`.

## Model environments

The two DiT snapshots contain identical environment files; the two FlowFormer snapshots contain equivalent flow environment files. They are intentionally retained in place:

```text
implementations/dit_unidirectional/environment.yml
implementations/dit_bidirectional/environment.yml
implementations/flowformer_unidirectional/environment.yml
implementations/flowformer_bidirectional/environment.yml
```

Create separate environments when exact experiment isolation matters. For example:

```bash
conda env create -n pde-carma-dit -f implementations/dit_unidirectional/environment.yml
conda env create -n pde-carma-flow -f implementations/flowformer_unidirectional/environment.yml
```

The supplied specifications are incomplete as standalone runtime environments:

- all source code imports PyTorch, but the YAML files do not install it;
- data readers import NumPy and xarray; xarray is absent from the DiT YAML;
- some plotting/reproduction utilities import SciPy;
- the source READMEs prescribe a machine-specific accelerator setup that is not a portable dependency guarantee.

Install a PyTorch build compatible with the actual operating system, accelerator, and driver. Do not assume a universal CUDA version. Then ensure at least the imports used by the selected workflow are available:

```bash
python -c "import torch, numpy, xarray, matplotlib"
```

Additional source environment entries include NetCDF/HDF5 support, pandas, openpyxl, scikit-learn, and plotting/reproduction utilities. Retain the source environment when reproducing historical runs; remove or pin packages only in a separately reviewed environment, never by editing the implementation copy.

## Installation verification

These checks do not require model dependencies:

```bash
python tools/verify_source_integrity.py
python scripts/run.py list
python scripts/run.py --dry-run infer flowformer-bidirectional -- --ckpt example.pth --num_ode_steps 5
```

Source `--help` requires the imports performed at the top of each original entry point, so it may fail until that model environment is complete. The wrapper's `--dry-run` is the dependency-free way to validate entry-point resolution and argument forwarding.

