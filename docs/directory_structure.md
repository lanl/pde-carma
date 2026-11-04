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
```

## Source-root mapping

| Archive | Archive root | Repository destination | Required training entry point |
|---|---|---|---|
| `pde-dit-adaln-unidirectional.zip` | `pde-dit-direction-specific-training-adaln/` | `implementations/dit_unidirectional/` | `train_dit_unidirectional.py` |
| `pde-dit-adaLN-bidirectional.zip` | `pde-dit-adaLN-zero/` | `implementations/dit_bidirectional/` | `train_dit.py` |
| `pde-flowformer-unidirectional.zip` | `pde-flowformer-direction-specific-training/` | `implementations/flowformer_unidirectional/` | `train_flowformer.py --direction forward|backward` |
| `pde-flowformer-bidirectional.zip` | `pde-flowformer-bidirectional-flowformer/` | `implementations/flowformer_bidirectional/` | `train_flowformer.py` |

The contents of each archive root were copied directly into its destination. No archive-root directory was inserted as an extra level.