# -*- coding: utf-8 -*-
# flake8: noqa: E501

import numpy as np

from pipelines.precipitation_model.impa.src.data.HDFDatasetMultiple import (
    HDFDatasetMultiple,
)
from pipelines.precipitation_model.impa.src.utils.dataframe_utils import (
    N_AFTER,
    N_BEFORE,
)

# For Rio de Janeiro only

elevation_file_small = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_rio_de_janeiro-res=2km-256x256.npy"
elevation_file_large = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_rio_de_janeiro-res=4km-256x256.npy"

latlon_file_small = (
    "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy"
)
latlon_file_large = (
    "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=4km-256x256.npy"
)


class HDFDatasetMerged(HDFDatasetMultiple):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Dataset filepath.
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        sat_dataframe,
        radar_dataframe,
        n_before=N_BEFORE,
        n_after=N_AFTER,
        x_transform=None,
        y_transform=None,
        leadtime_conditioning=False,
        predict_sat=False,
    ):
        dataframe_filepaths_array = np.empty((1, 2), dtype=object)
        elevation_filepaths_array = np.empty((1, 2), dtype=object)
        latlon_filepaths_array = np.empty((1, 2), dtype=object)

        dataframe_filepaths_array[0, 0] = radar_dataframe
        dataframe_filepaths_array[0, 1] = sat_dataframe
        elevation_filepaths_array[0, 0] = elevation_file_small
        elevation_filepaths_array[0, 1] = elevation_file_large
        latlon_filepaths_array[0, 0] = latlon_file_small
        latlon_filepaths_array[0, 1] = latlon_file_large

        n_before_array = np.array([n_before, n_before])
        n_after_array = np.array([n_after, n_after * predict_sat])
        n_before_resolution_array = np.array([1, 2])
        n_after_resolution_array = np.array([1, 2])

        super().__init__(
            dataframe_filepaths_array,
            elevation_filepaths_array,
            latlon_filepaths_array,
            n_before_array,
            n_after_array,
            n_before_resolution_array,
            n_after_resolution_array,
            x_transform,
            y_transform,
            leadtime_conditioning,
        )
