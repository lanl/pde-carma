def data_config(dataset_root=None):
    def p(rel_path):
        if dataset_root:
            return f"{dataset_root}/{rel_path}"
        return rel_path

    config = {
        "ce-rm": {
            "dataset_name": "ce-rm",
            "dataset_loc": p("datasets/pdegym/CE-RM/CE-RM.nc"),
            "var_name": "solution",
            "num_traj_use": None,
            "channels": 4,
            "img_size": (128, 128),
            "train_samples": 1030,
            "val_samples": 100,
            "test_samples": 130,
        },
        "ce-rpui": {
            "dataset_name": "ce-rpui",
            "dataset_loc": p("datasets/pdegym/CE-RPUI/CE-RPUI.nc"),
            "var_name": "data",
            "num_traj_use": None,
            "channels": 4,
            "img_size": (128, 128),
            "train_samples": 9640,
            "val_samples": 120,
            "test_samples": 240,
        },
        "ns-pwc": {
            "dataset_name": "ns-pwc",
            "dataset_loc": p("datasets/pdegym/NS-PwC/NS-PwC.nc"),
            "var_name": "velocity",
            "num_traj_use": None,
            "channels": 2,  # drop passive-tracer channel for training
            "img_size": (128, 128),
            "train_samples": 19640,
            "val_samples": 120,
            "test_samples": 240,
        },
        "2d-reacdiff": {
            "dataset_name": "2d-reacdiff",
            "dataset_loc": p("datasets/pdebench/vids_train_2d-reacdiff.npy"),
            "var_name": None,
            "num_traj_use": None,
            "channels": 2,
            "img_size": (128, 128),
            "train_samples": 800,
            "val_samples": 100,
            "test_samples": 100,
        },
        "2d-swe": {
            "dataset_name": "2d-swe",
            "dataset_loc": p("datasets/pdebench/vids_train_2d-swe.npy"),
            "var_name": None,
            "num_traj_use": None,
            "channels": 1,
            "img_size": (128, 128),
            "train_samples": 800,
            "val_samples": 100,
            "test_samples": 100,
        },
        "fns-kf": {
            "dataset_name": "fns-kf",
            "dataset_loc": (
                f"{dataset_root}/datasets/pdegym/FNS-KF/FNS-KF.nc" if dataset_root
                else "datasets/pdegym/FNS-KF/FNS-KF.nc"
            ),
            "var_name": "solution",
            "num_traj_use": None,
            "channels": 2,
            "img_size": (128, 128),
            "train_samples": 19640,
            "val_samples": 120,
            "test_samples": 240,
        },
    }
    return config
