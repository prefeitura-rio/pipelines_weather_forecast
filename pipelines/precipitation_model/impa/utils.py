# -*- coding: utf-8 -*-
# pylint: disable=C0103, C0302
"""
Utils
"""
import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple

import boto3  # pylint: disable=E0611, E0401
import h5py
import matplotlib.pyplot as plt  # pylint: disable=E0611, E0401
import numpy as np
import pandas as pd
from botocore import UNSIGNED  # pylint: disable=E0611, E0401
from botocore.config import Config  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.impa.src.utils.eval_utils import get_img


def download_file_from_s3(
    product, year, day_of_year, hour, download_base_path: str = "data/raw/satellite"
) -> str:
    """
    Download satellite data from AWS S3 bucket.

    Parameters
    ----------
    s3 : botocore.client.S3
        S3 client.
    product : str
        Product name (e.g. ABI-L2-RRQPEF).
    year : int
        Year.
    day_of_year : int
        Day of year (1-365).
    hour : int
        Hour of day (0-23).

    Returns
    -------
    None
    """
    # Initialize the S3 client
    signature_version = None
    if isinstance(UNSIGNED, type):
        # We're getting the class, not an instance
        signature_version = UNSIGNED()
    else:
        # We're getting an instance
        signature_version = UNSIGNED

    s3 = boto3.client("s3", config=Config(signature_version=signature_version))

    BUCKET_NAME = "noaa-goes16"
    # create parent folders
    prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
    parent_folder = Path(f"{download_base_path}/{prefix}")
    parent_folder.mkdir(parents=True, exist_ok=True)

    # download files
    # log(f"Bucket name = {BUCKET_NAME} and prefix = {prefix}")
    s3_result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/")
    for obj in s3_result.get("Contents", []):
        key = obj["Key"]
        file_name = key.split("/")[-1].split(".")[0]
        filepath = Path(f"{download_base_path}/{prefix}/{file_name}.nc")
        if filepath.exists():
            continue
        s3.download_file(BUCKET_NAME, key, filepath)
        log(f"Downloaded {key}")


def get_start_datetime(start_datetime=None):
    """
    Retorna um objeto datetime baseado no argumento fornecido.
    Se nenhum argumento for fornecido, retorna o datetime atual em UTC.

    Args:
        start_datetime (str): String de datetime no formato "YYYY-mm-dd HH:mm:ss" ou None.

    Returns:
        datetime.datetime: Objeto datetime no timezone UTC.
    """
    if start_datetime is None:
        dt = datetime.datetime.now(tz=datetime.timezone.utc)
    else:
        try:
            # Converte a string "YYYY-mm-dd HH:mm:ss" para datetime
            dt = datetime.datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=datetime.timezone.utc)  # Ajusta para UTC
        except ValueError as e:
            raise ValueError(
                f"Formato inválido: {start_datetime}. Use 'YYYY-mm-dd HH:mm:ss'"
            ) from e

    print(f"Running predictions on datetime [{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC]")
    return dt


def get_relevant_dates_informations(dt=None, n_historical_hours: int = 6) -> Tuple[List, List]:
    """
    This function calculates the relevant dates and their information.

    Args:
        dt (datetime.datetime, optional): The datetime for which the relevant dates are to be
        calculated.
        n_historical_hours (int, optional): The number of historical hours to consider.

    Returns:
        Tuple[List, List]: A tuple containing two lists. The first list contains the relevant
        datetimes, and the second list contains the relevant times separated into year, day
        of year, and hour.
    """
    relevant_dts = []

    timestep = 10
    dt_rounded = datetime.datetime(
        dt.year, dt.month, dt.day, dt.hour, dt.minute - dt.minute % timestep
    )
    # we want data from the last 6 hours, but we will more because sometimes there is a delay and
    # for ACHAF we need need more time in advance
    for i in range((60 * (n_historical_hours + 2)) // timestep):
        relevant_dts.append(dt_rounded - i * datetime.timedelta(minutes=timestep))

    relevant_times = set(
        (relevant_time.year, relevant_time.timetuple().tm_yday, relevant_time.hour)
        for relevant_time in relevant_dts
    )
    log(f"Relevant date hours to be used: {relevant_times}")
    return relevant_dts, sorted(relevant_times, key=lambda x: (-x[0], -x[1], -x[2]))


def get_filenames_to_process(
    product,
    download_base_path: str = "pipelines/precipitation_model/impa/data/raw/satellite",
) -> Tuple[List[str], List[str], bool]:
    """
    Get filenames to be processed
    """

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

    files = set()
    files = files.union(glob(f"{download_base_path}/{product}/*/*/*/*.nc"))
    files = list(files)
    files.sort()
    log(f">>> files\n {files}")
    log(f"First and last filename to be processed: {files[0]} {files[-1]}")
    return files, bands, include_dataset_name


def concat_processed_satellite(
    path,
    product,
) -> bool:
    """
    Concatenates processed satellite data files into a single dataframe.

    This function concatenates all CSV files in the specified path into a single pandas dataframe.
    It then resets the index of the dataframe and saves it to a feather file in a specified output
    path.

    Args:
        path (str): The path to the directory containing the CSV files to be concatenated.
        product (str): The product name to be used in the output file path.

    Returns:
        bool: Returns True if the concatenation and saving process is successful.
    """
    log(f"Start concating {product} processed files")
    dataframe = pd.concat(map(pd.read_csv, glob(path + "/*.csv")))
    dataframe.reset_index(drop=True, inplace=True)
    output_path = Path(f"pipelines/precipitation_model/impa/data/processed/satellite/{product}")
    output_path.mkdir(exist_ok=True, parents=True)
    dataframe.to_feather(f"{output_path}/SAT-real_time.feather")
    log(f"End concating {product} processed files")
    return True


# flake8: noqa: C901
def task_lag_img(
    lag: int,
    last_obs,
    last_obs_dt,
    timestep,
    data_source: str,
    model_names,
    preds,
    latlons,
    feature,
):

    BG_COLOR = "white"
    HEIGHT = 500
    WIDTH = 500

    future_dt = last_obs_dt + datetime.timedelta(minutes=lag * timestep)
    future_dt_str = future_dt.strftime("%Y-%m-%d %H:%M:%S")
    output_filepath = Path(
        f"pipelines/precipitation_model/impa/eval/viz/test/plot-real_time-{data_source}/lag={lag}"
    )

    future_imgs = len(preds) * [np.zeros((HEIGHT, WIDTH, 4), dtype=int)]

    # Treat data for Predict future
    for i, model_name in enumerate(model_names):
        # pred = preds[i]
        pred = h5py.File(preds[i])
        if i == 0:
            future_key = last_obs
        else:
            future_key = future_dt.strftime("%Y%m%d-%H%M")
            future_key = f"{last_obs}/{future_key}"
        try:
            if i == 0:
                values = np.array(pred[future_key]).reshape((*latlons.shape[:2], -1))[:, :, 0]
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    "Last observation",
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                    background=True,
                )
            else:
                values = np.array(pred[future_key])
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    model_name,
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                    background=True,
                )
        except KeyError:
            pass

    # Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)
    imgs = future_imgs
    filenames = []

    for j in range(len(preds)):
        log(f"Start creating image for {model_names[j]}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(imgs[j])
        if j == 0:
            ax.set_facecolor((0.1, 0.2, 0.5))
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        else:
            ax.axis("off")
        ax.axis("tight")

        fig.tight_layout()

        fig.subplots_adjust(bottom=0.1, top=0.9)
        save_path = Path(output_filepath / model_names[j])
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"{future_dt_str}.png"
        filenames.append(str(filename))
        log(f"\nFigure saved on {filename}. This prediction is for {future_dt}\n")
        fig.savefig(filename, transparent=True)
        fig.clf()
        plt.close()
    return filenames
