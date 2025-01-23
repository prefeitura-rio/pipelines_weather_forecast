# -*- coding: utf-8 -*-
# flake8: noqa: E501

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data

from pipelines.precipitation_model.impa.src.utils.dataframe_utils import (
    fetch_future_datetimes,
    fetch_reversed_past_datetimes,
)
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys


class HDFDatasetMultiple(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Dataset filepath.
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        dataframe_filepaths_array,
        elevation_filepaths_array,
        latlon_filepaths_array,
        n_before_array,
        n_after_array,
        n_before_resolution_array=None,
        n_after_resolution_array=None,
        x_transform=None,
        y_transform=None,
        leadtime_conditioning=False,
    ):
        if n_before_resolution_array is None:
            n_before_resolution_array = 2 * np.ones(n_before_array.shape, dtype=int)
        if n_after_resolution_array is None:
            n_after_resolution_array = 2 * np.ones(n_after_array.shape, dtype=int)
        self.n_before_resolution_array = n_before_resolution_array
        self.n_after_resolution_array = n_after_resolution_array

        assert n_before_array.dtype == int
        assert n_after_array.dtype == int
        assert n_before_array.ndim == 1
        assert len(n_before_array) == len(n_after_array)
        assert n_before_resolution_array.dtype == int
        assert n_after_resolution_array.dtype == int
        assert len(n_before_resolution_array) == len(n_before_array)
        assert len(n_after_resolution_array) == len(n_after_array)
        assert len(n_before_array) == dataframe_filepaths_array.shape[1]
        assert (n_before_array >= 0).all()
        assert (n_after_array >= 0).all()
        assert n_before_array[0] > 0
        assert n_after_array[0] > 0

        assert elevation_filepaths_array.shape[0] == dataframe_filepaths_array.shape[0]
        assert elevation_filepaths_array.shape[1] == 2
        assert latlon_filepaths_array.shape[0] == dataframe_filepaths_array.shape[0]
        assert latlon_filepaths_array.shape[1] == 2

        assert not leadtime_conditioning or n_after_array[0] == n_after_array.sum()

        self.x_transform = x_transform
        self.y_transform = y_transform
        self.leadtime_conditioning = leadtime_conditioning

        new_dataframe_filepaths_array = dataframe_filepaths_array.copy().flatten()
        for i, filepath in enumerate(dataframe_filepaths_array.flatten()):
            filepath = Path(filepath).resolve()
            shm_path = str(filepath).replace(str(filepath.parents[4]), "/dev/shm")
            if Path(shm_path).exists():
                # print_ok("File found in /dev/shm, using it.")
                new_dataframe_filepaths_array[i] = shm_path
            else:
                # print_warning("File not found in /dev/shm, using original path.")
                new_dataframe_filepaths_array[i] = str(filepath)

        self.dataframe_shm_filepaths_array = new_dataframe_filepaths_array.reshape(
            dataframe_filepaths_array.shape
        )
        self.dataframe_filepaths_array = dataframe_filepaths_array
        self.n_before_array = n_before_array
        self.n_after_array = n_after_array
        self.elevation_tensors = []
        self.latlon_tensors = []
        for elevation_filepath_small, elevation_filepath_large in elevation_filepaths_array:
            el1 = torch.tensor(np.load(elevation_filepath_small))
            el2 = torch.tensor(np.load(elevation_filepath_large))
            el = torch.stack((el1, el2), dim=-1).type(torch.float32)
            self.elevation_tensors.append(el)

        for latlon_filepath_small, latlon_filepath_large in latlon_filepaths_array:
            latlon1 = torch.tensor(np.load(latlon_filepath_small))
            latlon2 = torch.tensor(np.load(latlon_filepath_large))
            latlon = torch.stack((latlon1, latlon2), dim=-1).type(torch.float32)
            self.latlon_tensors.append(latlon)
        self._load_keys()

    def _load_keys(self):
        self.ni = None
        self.nj = None
        timestep = None
        self.subgrid_i1 = None
        self.subgrid_i2 = None
        self.subgrid_j1 = None
        self.subgrid_j2 = None

        self.ds_indices = []
        self.keys = np.array([])
        self.past_keys = np.array([])
        self.future_keys = np.array([])
        for i in range(self.dataframe_shm_filepaths_array.shape[0]):
            for j in range(self.dataframe_shm_filepaths_array.shape[1]):
                filepath = self.dataframe_shm_filepaths_array[i, j]
                with h5py.File(filepath) as hdf:
                    what = hdf["what"]
                    if i == 0:
                        timestep = int(what.attrs["timestep"])
                        self.ni = what.attrs["ni"]
                        self.nj = what.attrs["nj"]
                    else:
                        assert timestep == int(what.attrs["timestep"])
                        assert self.ni == what.attrs["ni"]
                        assert self.nj == what.attrs["nj"]

                    if "split_info/split_datetime_keys" in hdf:
                        if j == 0:
                            new_keys = set(hdf["split_info"]["split_datetime_keys"])
                        else:
                            new_keys = set(hdf["split_info"]["split_datetime_keys"]).intersection(
                                new_keys
                            )
                    else:
                        if j == 0:
                            new_keys = set(get_dataset_keys(hdf))
                        else:
                            new_keys = set(get_dataset_keys(hdf)).intersection(new_keys)

            new_keys = sorted(list(new_keys))
            new_keys = np.array(new_keys).astype(str)
            self.keys = np.append(self.keys, new_keys)
            try:
                self.past_keys = np.vstack(
                    [
                        self.past_keys,
                        np.array(
                            fetch_reversed_past_datetimes(
                                new_keys, self.n_before_array.max(), timestep
                            )
                        ),
                    ]
                )
            except ValueError:
                self.past_keys = np.array(
                    fetch_reversed_past_datetimes(new_keys, self.n_before_array.max(), timestep)
                )
            try:
                self.future_keys = np.vstack(
                    [
                        self.future_keys,
                        np.array(
                            fetch_future_datetimes(new_keys, self.n_after_array.max(), timestep)
                        ),
                    ]
                )
            except ValueError:
                self.future_keys = np.array(
                    fetch_future_datetimes(new_keys, self.n_after_array.max(), timestep)
                )
            try:
                self.ds_indices.append(len(new_keys) + self.ds_indices[-1])
            except IndexError:
                self.ds_indices.append(len(new_keys))

    def _get_hdf_index(self, index):
        for i, ds_index in enumerate(self.ds_indices):
            if index < ds_index:
                return i

    def __getitem__(self, index):
        X = (
            torch.ones(
                (self.ni, self.nj, (self.n_before_array * self.n_before_resolution_array).sum())
            )
            * torch.inf
        )
        if self.leadtime_conditioning:
            pass
        else:
            Y = (
                torch.ones(
                    (self.ni, self.nj, (self.n_after_array * self.n_after_resolution_array).sum())
                )
                * torch.inf
            )

        if self.leadtime_conditioning:
            leadtime_index = index % self.n_after_array[0]
            index = index // self.n_after_array[0]

        hdf_index = self._get_hdf_index(index)
        filepaths = self.dataframe_shm_filepaths_array[hdf_index]
        for j, filepath in enumerate(filepaths):
            with h5py.File(filepath, "r") as hdf:
                # for i, key in enumerate(self.past_keys[index]):
                n_resolution = self.n_before_resolution_array[j]
                for i in range(self.n_before_array[j]):
                    tensor_ind = (
                        np.cumsum(self.n_before_array * self.n_before_resolution_array)[j]
                        - i * n_resolution
                        - n_resolution
                    )
                    try:
                        X[:, :, tensor_ind : tensor_ind + n_resolution] = torch.as_tensor(
                            np.array(hdf[self.past_keys[index][i]]).reshape(
                                (self.ni, self.nj, n_resolution)
                            )
                        )
                    except KeyError:
                        X[:, :, tensor_ind : tensor_ind + n_resolution] = (
                            torch.ones((self.ni, self.nj, 1)) * np.nan
                        )
                if self.leadtime_conditioning and j == 0:
                    try:
                        Y = torch.as_tensor(
                            np.array(hdf[self.future_keys[index][leadtime_index]])
                        ).reshape((self.ni, self.nj, self.n_after_resolution_array[0]))
                    except KeyError:
                        Y = (
                            torch.ones((self.ni, self.nj, self.n_after_resolution_array[0]))
                            * np.nan
                        )
                else:
                    # for i, key in enumerate(self.future_keys[index]):
                    n_resolution = self.n_after_resolution_array[j]
                    for i in range(self.n_after_array[j]):
                        if j == 0:
                            cumsum = 0
                        else:
                            cumsum = np.cumsum(self.n_after_array * self.n_after_resolution_array)[
                                j - 1
                            ]
                        tensor_ind = cumsum + i * n_resolution
                        try:
                            Y[:, :, tensor_ind : tensor_ind + n_resolution] = torch.as_tensor(
                                np.array(hdf[self.future_keys[index][i]])
                            ).reshape((self.ni, self.nj, n_resolution))

                        except KeyError:
                            Y[:, :, tensor_ind : tensor_ind + n_resolution] = (
                                torch.ones((self.ni, self.nj, 1)) * np.nan
                            )
        if self.x_transform:
            X = self.x_transform(X)
        if self.y_transform:
            Y = self.y_transform(Y)

        ### Metadata
        date = self.past_keys[index][-1]
        # year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        hour = int(date[9:11])
        minute = int(date[11:13])
        date = torch.tensor(
            [month / 12, day / 31, hour / 24, minute / 60], dtype=torch.float32
        ).reshape((1, 1, -1))
        date_tensor = date.expand((self.ni, self.nj, -1))

        metadata_tensor = torch.cat(
            (
                self.elevation_tensors[hdf_index],
                self.latlon_tensors[hdf_index].reshape(self.ni, self.nj, -1),
                date_tensor,
            ),
            dim=-1,
        )
        if self.leadtime_conditioning:
            return (X, Y, metadata_tensor, leadtime_index)
        else:
            return (X, Y, metadata_tensor)

    def __len__(self):
        if self.leadtime_conditioning:
            return len(self.keys) * self.n_after_array[0]
        else:
            return len(self.keys)

    def get_sample_weights(self, min_sum=0):
        # Weights are only calculated with respect to the first column dataframes
        weights = None
        for filepath in self.dataframe_filepaths_array[:, 0]:
            filepath = Path(filepath)
            stem = filepath.stem
            weights_filepath = filepath.parents[0] / f"{stem}_sample_weights.npy"
            inds_filepath = filepath.parents[0] / f"{stem}_split_datetime_inds.npy"
            pre_weights = np.load(weights_filepath)
            pre_weights_size = len(pre_weights)
            pre_weights = np.append(pre_weights, np.array([pre_weights.min()]))
            inds = np.load(inds_filepath).reshape(-1, 1)
            deltas = np.arange(
                -self.n_before_array[0] + 1, self.n_after_array[0] + 1, dtype=int
            ).reshape(1, -1)
            slices = inds + deltas
            slices[np.logical_or(slices < 0, slices >= pre_weights_size)] = pre_weights_size
            summed_weights = pre_weights[slices].sum(axis=1).reshape(-1)

            if weights is None:
                weights = summed_weights
            else:
                weights = np.append(weights, summed_weights)
        if self.leadtime_conditioning:
            weights = np.tile(weights, (self.n_after_array[0], 1)).T.flatten()
        min_sum = max(min_sum, -weights.min())
        weights = weights + min_sum
        assert len(weights) == len(self)
        return weights / weights.sum()
