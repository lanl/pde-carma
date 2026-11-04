# Datasets

Dataset can be downloaded from pdebench and pdegym. 
The table below reports configured support in the four unchanged `utils/data_config.py` files.

| Dataset | Uni-DiT | Bi-DiT | Uni-FlowFormer | Bi-FlowFormer |
|---|:---:|:---:|:---:|:---:|
| CE-RM | yes | yes | yes | yes |
| CE-RPUI | no | yes | no | yes |
| NS-PwC | no | yes | no | yes |
| 2D-SWE | yes | yes | yes | yes |
| FNS-KF | no | yes | no | yes |
| NS-SL | no | yes | no | yes |

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

NetCDF readers use the source-configured variable names (`solution`, `data`, or `velocity`) and expect trajectory data compatible with `[sample,time,channel,y,x]`. 

## Configured splits and channels

| Dataset | Channels used | Train | Validation | Test |
|---|---:|---:|---:|---:|
| CE-RM | 4 | 1,030 | 100 | 130 |
| CE-RPUI | 4 | 9,640 | 120 | 240 |
| NS-PwC | 2 | 19,640 | 120 | 240 |
| 2D-SWE | 1 | 800 | 100 | 100 |
| FNS-KF | 2 | 19,640 | 120 | 240 |
| NS-SL | 2 | 39,640 | 120 | 240 |
