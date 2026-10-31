import xarray as xr
import numpy as np
import torch

def data_arr_nc(dataset_loc, var_name):
    with xr.open_dataset(dataset_loc) as ds:
        da = ds[var_name]  # xarray.DataArray
        print("DataArray dims in file:", da.dims)  
        da = da.transpose("sample", "time", "channel", "y", "x")  
        arr = da.values.astype(np.float32)
        return arr

def data_arr_npy(path):
    arr = np.load(path).astype(np.float32)
    return arr