# -*- coding: utf-8 -*-
import h5py


def get_dataset_keys(f):
    keys = []
    for date in f.keys():
        if date in ["what", "subgrid", "split_info"]:
            continue
        g = f[date]
        g.visit(
            lambda key: keys.append(f"{date}/{key}") if isinstance(g[key], h5py.Dataset) else None
        )
    return keys


def array_to_pred_hdf(arr, keys, future_keys, output_filepath):
    with h5py.File(output_filepath, "a") as hdf:
        for i, key in enumerate(keys):
            try:
                key = key.decode("utf-8")
            except AttributeError:
                pass

            for j, future_key in enumerate(future_keys[i]):
                try:
                    future_key = future_key.decode("utf-8")
                except AttributeError:
                    pass
                pred_key = f"{key}/{future_key.replace('/','-')}"
                hdf.create_dataset(pred_key, data=arr[i, :, :, j], compression="lzf")
