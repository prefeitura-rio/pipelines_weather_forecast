# -*- coding: utf-8 -*-
# flake8: noqa: E501

from argparse import ArgumentParser
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pyproj import Proj
from tqdm import tqdm


def process_file(
    file_path: str,
    bands: list[str],
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    include_dataset_name: bool = False,
) -> pd.DataFrame:
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
    # Read satellite data
    dataset = xr.open_dataset(file_path)

    # Retrieve datetimes of file creation and start and end of scan (UTC)
    scan_start = datetime.strptime(dataset.time_coverage_start, "%Y-%m-%dT%H:%M:%S.%fZ")
    scan_end = datetime.strptime(dataset.time_coverage_end, "%Y-%m-%dT%H:%M:%S.%fZ")
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
        np.array(data).reshape(len(data), -1).T,
        columns=["lat", "lon", *bands],
    )

    # Remove nan and infinite values
    df = df.replace(np.inf, np.nan)
    df = df.dropna(how="any", axis=0)

    # Discard observations for latitudes and longitudes outside bounds
    if lat_bounds:
        df = df[(df["lat"] > lat_bounds[0]) & (df["lat"] < lat_bounds[1])]
    if lon_bounds:
        df = df[(df["lon"] > lon_bounds[0]) & (df["lon"] < lon_bounds[1])]

    # Include datetimes of file creation and start and end of scan (UTC-3)
    df["start"] = scan_start  # - timedelta(hours=3)
    df["end"] = scan_end  # - timedelta(hours=3)
    df["creation"] = creation  # - timedelta(hours=3)

    if include_dataset_name:
        df["name"] = "_".join(dataset.dataset_name.split("_")[:2])

    return df


def process_satellite(
    product="ABI-L2-RRQPEF", lat_min=-26.0, lat_max=-19.0, lon_min=-47.0, lon_max=-40.0, num_workers=16, day=-1, year=-1
):
    match product:
        case "ABI-L2-MCMIPF":  # Cloud and Moisture Imagery
            bands = ["CMI_C08", "CMI_C09", "CMI_C10", "CMI_C11"]
            include_dataset_name = False
        case "ABI-L2-RRQPEF":  # Rainfall Rate (Quantitative Precipitation Estimate)
            bands = ["RRQPE"]
            include_dataset_name = False
        case "ABI-L2-DMWF":
            bands = [
                "wind_direction",
                "wind_speed",
                "temperature",
                "pressure",
            ]
            include_dataset_name = True
        case "ABI-L2-ACHAF":
            bands = ["HT"]
            include_dataset_name = False
        case _:
            raise ValueError("Unsupported product selected.")

    lat_bounds = lat_min, lat_max
    lon_bounds = lon_min, lon_max

    def load_entire_day(ts: pd.Timestamp) -> pd.DataFrame:
        year = ts.year
        day = ts.dayofyear
        return pd.concat(
            Parallel(n_jobs=num_workers)(
                delayed(process_file)(file, bands, lat_bounds, lon_bounds, include_dataset_name)
                for file in tqdm(glob(f"data/raw/satellite/{product}/{year}/{day:03d}/*/*.nc"))
            )
        )

    end_date = datetime(year, 1, 1) + timedelta(day - 1)
    today_file = Path(f"data/processed/satellite/{product}/{end_date.date()}.feather")
    if today_file.is_file():
        # Do not process older dates
        start_date = end_date
    else:
        start_date = datetime(year, 1, 1) + timedelta(day - 4)

    df_current = load_entire_day(pd.Timestamp(start_date))
    output_path = Path(f"data/processed/satellite/{product}")
    output_path.mkdir(exist_ok=True, parents=True)

    for date in tqdm(
        pd.date_range(start_date, end_date),
        desc="Saving files",
    ):
        next_date = date + timedelta(days=1)
        try:
            df_next = load_entire_day(next_date)
            df = pd.concat([df_current, df_next])
        except ValueError:
            df = df_current
            df_next = None
        df = df[df["creation"].dt.date == date.date()]
        df = df.reset_index(drop=True)
        df.to_feather(f"{output_path}/{date.date()}.feather")
        try:
            df_current = df_next.copy()
        except AttributeError:
            pass
        del df_next


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--product", type=str, default="ABI-L2-RRQPEF")
    parser.add_argument("--lat_min", type=float, default=-26.0)
    parser.add_argument("--lat_max", type=float, default=-19.0)
    parser.add_argument("--lon_min", type=float, default=-47.0)
    parser.add_argument("--lon_max", type=float, default=-40.0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--day", type=int, default=-1)
    parser.add_argument("--year", type=int, default=-1)
    args = parser.parse_args()

    process_satellite(**vars(args))