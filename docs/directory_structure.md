# Repository structure

The supplied `directory_structure.md` was used as a responsibility guide, but the user's later preservation requirement takes precedence over its proposed `src/carma` refactor. The actual repository keeps all source files unchanged under `implementations/` and places every new facility outside that boundary.

```text
pde-carma/
├── README.md
├── .gitattributes                  # Disable EOL normalization in implementations/
├── pyproject.toml
├── configs/reproduction.yaml       # Documentation-only paper settings
├── implementations/
│   ├── dit_unidirectional/          # 43 files from the Uni-DiT archive
│   ├── dit_bidirectional/           # 40 files from the Bi-DiT archive
│   ├── flowformer_unidirectional/   # 38 files from the Uni-FlowFormer archive
│   └── flowformer_bidirectional/    # 49 files from the Bi-FlowFormer archive
├── src/
│   └── pde_carma/                   # Registry, command resolution, subprocess launcher
├── scripts/
│   ├── run.py                       # Unified train/infer/evaluate/CARMA/reproduce CLI
│   ├── train/                       # Four convenience launchers
│   └── infer/                       # Four convenience launchers
├── tools/
│   ├── check_source_syntax.py       # Parse original Python without importing it
│   ├── smoke_data_loaders.py         # Synthetic normalization/pairing checks
│   ├── smoke_models.py              # Isolated synthetic model/runtime checks
│   └── verify_source_integrity.py   # Manifest/path/hash verification
├── tests/
│   ├── test_data_smoke_tool.py       # Dependency-light data-smoke behavior
│   ├── test_model_smoke_tool.py      # Dependency-light smoke-tool behavior
│   ├── test_orchestration.py        # Dependency-light wrapper tests
│   └── test_source_integrity.py     # Full 170-file manifest check
├── datasets/                        # Local-only inputs
├── artifacts/                       # Local-only external outputs
└── docs/
    ├── directory_structure.md
    ├── installation.md
    ├── datasets.md
    ├── training.md
    ├── inference.md
    ├── reproducibility.md
    ├── source_manifest.json
    ├── source_integrity.md
    ├── validation_report.md
    └── publication_readiness.md
```

## Source-root mapping

| Archive | Archive root | Repository destination | Required training entry point |
|---|---|---|---|
| `pde-dit-adaln-unidirectional.zip` | `pde-dit-direction-specific-training-adaln/` | `implementations/dit_unidirectional/` | `train_dit_unidirectional.py` |
| `pde-dit-adaLN-bidirectional.zip` | `pde-dit-adaLN-zero/` | `implementations/dit_bidirectional/` | `train_dit.py` |
| `pde-flowformer-unidirectional.zip` | `pde-flowformer-direction-specific-training/` | `implementations/flowformer_unidirectional/` | `train_flowformer.py --direction forward|backward` |
| `pde-flowformer-bidirectional.zip` | `pde-flowformer-bidirectional-flowformer/` | `implementations/flowformer_bidirectional/` | `train_flowformer.py` |

The contents of each archive root were copied directly into its destination. No archive-root directory was inserted as an extra level.

## Immutability boundary

Everything below `implementations/` is source-of-truth material and must remain byte-for-byte identical to the extracted archives. In particular:

- original `README.md`, environment files, launchers, hardcoded paths, and duplicate files remain present;
- no `__init__.py` files, import rewrites, portability fixes, or formatting changes are inserted there;
- no shared module is substituted for an implementation-local module;
- source entry points continue to resolve `utils.*` exactly as before.

Run `python tools/verify_source_integrity.py` after any checkout or packaging operation. Any missing, extra, or changed implementation file is a release-blocking discrepancy.

The root `.gitattributes` marks `implementations/**` as `-text`, preventing Git from normalizing line endings and invalidating the byte-level hashes on another platform. The launcher also sets `PYTHONDONTWRITEBYTECODE=1` for child processes so normal imports do not add `__pycache__` files to the preserved trees.

## External orchestration boundary

`src/pde_carma` knows only implementation locations and original entry points. It executes a child Python process with its working directory set to the selected implementation. This preserves source-local imports and relative paths without changing the source.

The wrappers do not normalize flags, translate hyperparameters, merge checkpoint formats, or import model modules. They only select the original script and forward the remaining arguments in order.

## Generated files

Original scripts may create `models/`, `results/`, `datasets/`, plots, `.npy` arrays, and checkpoints inside an implementation tree. These generated files are not part of the immutable source manifest and should remain untracked. A pristine integrity check must be run against the committed source before release; use explicit data roots or external links where the source CLI supports them.

