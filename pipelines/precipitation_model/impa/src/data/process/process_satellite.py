# -*- coding: utf-8 -*-
"""
Process satellite data
"""
# flake8: noqa: E501
# pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-arguments
import gc
# from argparse import ArgumentParser  # aqui
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
# import pyspark.pandas as pd

import xarray as xr

from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401
from pyproj import Proj
from tqdm import tqdm  # pylint: disable=E0611, E0401


def process_file(
    product,
    file_path: str,
    bands: list[str],
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    include_dataset_name: bool = False,
    lat_min=-26.0,
    lat_max=-19.0,
    lon_min=-47.0,
    lon_max=-40.0,
) -> str:
    """Returns processed satellite data for desired region and bands.

    Conversion of coordinates to latitudes and longitudes based on [1].

    Args:
        file_path: path to netcdf file with satellite data
        bands: bands to be extracted from data
        lat_bounds: minimum and maximum latitudes to consider
        lon_bounds: minimum and maximum longitudes to consider

    References:
        [1] https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb
        [2] https://proj4.org/operations/projections/geos.html
    """

    # cpu_usage = psutil.cpu_percent(interval=1, percpu=True)  # Lista de uso de cada núcleo
    # cpu_usage_total = sum(cpu_usage) / len(cpu_usage)  # Média de uso de CPU (total)

    # # Uso total de memória do sistema
    # memory_info = psutil.virtual_memory()
    # total_memory = memory_info.total / (1024**3)  # Convertendo para GB
    # used_memory = memory_info.used / (1024**3)
    # free_memory = memory_info.available / (1024**3)

    # # Exibir os resultados
    # log(
    #     f"Uso total de CPU por núcleo (%): {cpu_usage}, Uso médio total de CPU (%): {cpu_usage_total:.2f}%"
    # )
    # log(
    #     f"Memória total: {total_memory:.2f} GB, Memória usada: {used_memory:.2f} GB, Memória livre: {free_memory:.2f} GB"
    # )
    lat_bounds = lat_min, lat_max
    lon_bounds = lon_min, lon_max

    # Read satellite data
    dataset = xr.open_dataset(file_path)

    # Retrieve datetimes of file creation and start and end of scan (UTC)
    # scan_start = datetime.strptime(dataset.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")
    # scan_end = datetime.strptime(dataset.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")
    creation = datetime.strptime(dataset.date_created, "%Y-%m-%dT%H:%M:%S.%fZ")

    if hasattr(dataset, "lon") and hasattr(dataset, "lat"):
        lons, lats = dataset["lon"], dataset["lat"]
    else:
        # Load satellite height, longitude and sweep
        sat_h = dataset["goes_imager_projection"].perspective_point_height
        sat_lon = dataset["goes_imager_projection"].longitude_of_projection_origin
        sat_sweep = dataset["goes_imager_projection"].sweep_angle_axis

        # Calculate projection x and y coordinates as the scanning angle (in radians)
        #   multiplied by the satellite height (cf. [2])
        x = dataset["x"][:] * sat_h
        y = dataset["y"][:] * sat_h

        # Create a pyproj geostationary map object
        p = Proj(proj="geos", h=sat_h, lon_0=sat_lon, sweep=sat_sweep)

        # Perform cartographic transformation, that is, convert
        #  image projection coordinates (x and y) to latitude and longitude values
        XX, YY = np.meshgrid(x, y)
        lons, lats = p(XX, YY, inverse=True)

    # Load data from bands of interest and append to latitude and longitude
    data = [lats, lons] + [dataset[band].data for band in bands]

    # Construct dataframe
    df = pd.DataFrame(
        np.array(data).astype(dtype=np.float32).reshape(len(data), -1).T,
        columns=["lat", "lon", *bands],
    )

    # Remove nan and infinite values
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(how="any", axis=0, inplace=True)

    # Discard observations for latitudes and longitudes outside bounds
    if lat_bounds:
        df.drop(df[(df["lat"] <= lat_bounds[0]) | (df["lat"] >= lat_bounds[1])].index, inplace=True)
    if lon_bounds:
        df.drop(df[(df["lon"] <= lon_bounds[0]) | (df["lon"] >= lon_bounds[1])].index, inplace=True)

    # Include datetimes of file creation and start and end of scan (UTC-3)
    # df["start"] = scan_start  # - timedelta(hours=3)
    # df["end"] = scan_end  # - timedelta(hours=3)
    df["creation"] = creation  # - timedelta(hours=3)

    if include_dataset_name:
        df["name"] = "_".join(dataset.dataset_name.split("_")[:2])
    gc.collect()

    output_path = Path(f"pipelines/precipitation_model/impa/data/processed_temp/satellite/{product}/")
    output_path.mkdir(exist_ok=True, parents=True)
    filename = file_path.split("/")[-1].split(".")[0] + ".csv"
    output_filename = output_path / filename
    df.to_csv(output_filename, index=False)
    return output_filename
