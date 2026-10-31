# Datasets

No dataset binaries are included. The table below reports configured support in the four unchanged `utils/data_config.py` files; it does not claim that acquisition, preprocessing, or training was validated.

| Dataset | Uni-DiT | Bi-DiT | Uni-FlowFormer | Bi-FlowFormer |
|---|:---:|:---:|:---:|:---:|
| CE-RM | yes | yes | yes | yes |
| CE-RPUI | yes | yes | yes | yes |
| NS-PwC | yes | yes | yes | yes |
| 2D-ReacDiff | yes | yes | yes | yes |
| 2D-SWE | yes | yes | yes | yes |
| FNS-KF | no | yes | yes | yes |
| NS-SL | no | yes | no | yes |
| Active Matter | no | yes | no | yes |

## Expected relative paths

The registries expect these layouts beneath a dataset root:

```text
datasets/
├── pdegym/
│   ├── CE-RM/CE-RM.nc
│   ├── CE-RPUI/CE-RPUI.nc
│   ├── NS-PwC/NS-PwC.nc
│   ├── FNS-KF/FNS-KF.nc
│   └── NS-SL/NS-SL.nc
├── pdebench/
│   ├── vids_train_2d-reacdiff.npy
│   ├── vids_val_2d-reacdiff.npy
│   ├── vids_test_2d-reacdiff.npy
│   ├── vids_train_2d-swe.npy
│   ├── vids_val_2d-swe.npy
│   └── vids_test_2d-swe.npy
└── thewell/
    ├── vids_train_active-matter.npy
    ├── vids_val_active-matter.npy
    └── vids_test_active-matter.npy
```

NetCDF readers use the source-configured variable names (`solution`, `data`, or `velocity`) and expect trajectory data compatible with `[sample,time,channel,y,x]`. The configured image sizes are 128×128 except Active Matter at 256×256.

## Configured splits and channels

| Dataset | Channels used | Train | Validation | Test |
|---|---:|---:|---:|---:|
| CE-RM | 4 | 1,030 | 100 | 130 |
| CE-RPUI | 4 | 9,640 | 120 | 240 |
| NS-PwC | 2 | 19,640 | 120 | 240 |
| 2D-ReacDiff | 2 | 800 | 100 | 100 |
| 2D-SWE | 1 | 800 | 100 | 100 |
| FNS-KF | 2 | 19,640 | 120 | 240 |
| NS-SL | 2 | 39,640 | 120 | 240 |
| Active Matter | 11 | 175 | 24 | 26 |

## Root-path behavior

Path portability is implementation-specific and remains unchanged:

| Workflow | `--dataset_root` in source CLI | Behavior |
|---|:---:|---|
| Uni-DiT train/infer | yes | Resolves `ROOT/datasets/...` |
| Bi-DiT train/infer | yes | Resolves `ROOT/datasets/...` |
| Uni-FlowFormer train | yes | Resolves `ROOT/datasets/...` |
| Uni-FlowFormer infer | no | Reads its implementation-local `datasets/` |
| Bi-FlowFormer train/infer | no | Uses implementation-local relative paths |

For workflows without a dataset-root flag, place or link `datasets/` beneath the selected implementation directory. This is an external filesystem arrangement, not a source edit. Keep local data untracked and run the integrity check on a pristine source tree before mounting or generating data.

## Preserved preprocessing behavior

- Pre-split NPY datasets compute normalization statistics from the training split and apply them to validation/test data.
- Single-file NC/NPY datasets compute channel statistics over the loaded trajectories before train/validation/test slicing. This leaks validation/test distribution information by current evaluation standards, but changing it would break historical numerical behavior, so it is preserved and must be disclosed in new results.
- Uni-FlowFormer reverses the full time axis for `--direction backward`, then constructs ordinary adjacent pairs.
- Bi-DiT and Bi-FlowFormer construct both adjacent directions and attach `[1,0]` or `[0,1]` controls.
- Normalization statistics are not stored in the source checkpoints.

The loaders write normalized validation/test arrays under their selected `datasets/` directory. Uni-FlowFormer writes direction-specific names such as `vids_test_<dataset>_backward.npy`; other variants use `vids_test_<dataset>.npy`.

## Availability and licensing

The snapshots do not provide complete, verified acquisition and redistribution instructions for all configured datasets. Before a public release or paper artifact:

1. identify the authoritative upstream source and version for every dataset;
2. record any conversion used to produce pre-split NPY arrays;
3. confirm variable/channel ordering and split provenance;
4. confirm redistribution and citation requirements;
5. keep dataset binaries out of Git.

