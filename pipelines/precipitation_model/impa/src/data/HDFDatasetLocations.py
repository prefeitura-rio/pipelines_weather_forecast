# -*- coding: utf-8 -*-
import numpy as np

from pipelines.precipitation_model.impa.src.data.HDFDatasetMultiple import (
    HDFDatasetMultiple,
)
from pipelines.precipitation_model.impa.src.utils.dataframe_utils import (
    N_AFTER,
    N_BEFORE,
)

elevation_file_small = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_{location}-res=2km-256x256.npy"
elevation_file_large = "pipelines/precipitation_model/impa/data/processed/elevations_data/elevation_{location}-res=4km-256x256.npy"

latlon_file_small = (
    "pipelines/precipitation_model/impa/data/dataframe_grids/{location}-res=2km-256x256.npy"
)
latlon_file_large = (
    "pipelines/precipitation_model/impa/data/dataframe_grids/{location}-res=4km-256x256.npy"
)


class HDFDatasetLocations(HDFDatasetMultiple):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Dataset filepath.
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        dataframe,
        locations,
        n_before=N_BEFORE,
        n_after=N_AFTER,
        x_transform=None,
        y_transform=None,
        leadtime_conditioning=False,
    ):
        dataframe_filepaths_array = np.empty((len(locations), 1), dtype=object)
        elevation_filepaths_array = np.empty((len(locations), 2), dtype=object)
        latlon_filepaths_array = np.empty((len(locations), 2), dtype=object)
        for i, location in enumerate(locations):
            dataframe_filepaths_array[i, 0] = dataframe.format(location=location)
            elevation_filepaths_array[i, 0] = elevation_file_small.format(location=location)
            elevation_filepaths_array[i, 1] = elevation_file_large.format(location=location)
            latlon_filepaths_array[i, 0] = latlon_file_small.format(location=location)
            latlon_filepaths_array[i, 1] = latlon_file_large.format(location=location)

        n_before_array = np.array([n_before])
        n_after_array = np.array([n_after])

        super().__init__(
            dataframe_filepaths_array,
            elevation_filepaths_array,
            latlon_filepaths_array,
            n_before_array,
            n_after_array,
            x_transform=x_transform,
            y_transform=y_transform,
            leadtime_conditioning=leadtime_conditioning,
        )
