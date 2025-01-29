# -*- coding: utf-8 -*-
# pylint: disable=import-error, invalid-name, missing-function-docstring, too-many-arguments
# flake8: noqa: E501
"""
Tasks
"""

import json
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Set, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import tqdm
from prefect import task  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401

from pipelines.precipitation_model.impa.src.data.process.build_dataframe_from_sat import (
    build_dataframe_from_sat,
)
from pipelines.precipitation_model.impa.src.data.process.process_satellite import (
    process_file,
)
from pipelines.precipitation_model.impa.src.eval.predict_real_time import predict
from pipelines.precipitation_model.impa.src.utils.general_utils import print_warning
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys
from pipelines.precipitation_model.impa.utils import (
    concat_processed_satellite,
    download_file_from_s3,
    get_filenames_to_process,
    get_relevant_dates_informations,
    get_start_datetime,
    task_lag_img,
)


@task
def get_start_datetime_task(start_datetime=None):
    """
    Retorna um objeto datetime baseado no argumento fornecido.
    Se nenhum argumento for fornecido, retorna o datetime atual em UTC.

    Args:
        start_datetime (str): String de datetime no formato "YYYY-mm-dd HH:mm:ss" ou None.

    Returns:
        datetime.datetime: Objeto datetime no timezone UTC.
    """
    return get_start_datetime(start_datetime=start_datetime)


@task(nout=2)
def get_relevant_dates_informations_task(dt=None, n_historical_hours: int = 6) -> Tuple[List, List]:
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
    return get_relevant_dates_informations(
        dt=dt,
        n_historical_hours=n_historical_hours,
    )


@task
def download_files_from_s3_task(
    product,
    relevant_times: Union[List, Set],
    download_base_path: str = "pipelines/precipitation_model/impa/data/raw/satellite",
    wait=None,  # pylint: disable=unused-argument
):
    """
    Download satellite data from AWS S3 bucket.

    Parameters
    ----------

    Returns
    -------
    None
    """
    relevant_times = [relevant_times] if not isinstance(relevant_times, list) else relevant_times
    log(f"\n\nDownloading relevant times for product {product}:\n{relevant_times}\n\n")
    for relevant_time in relevant_times:
        download_file_from_s3(product, *relevant_time, download_base_path)
    return True


@task(nout=3)
def get_filenames_to_process_task(
    product,
    download_base_path: str = "pipelines/precipitation_model/impa/data/raw/satellite",
    wait=None,  # pylint: disable=unused-argument
) -> Tuple[List[str], List[str], bool]:
    return get_filenames_to_process(
        product=product,
        download_base_path=download_base_path,
    )


@task
def process_satellite_task(
    file_paths: Union[List[str], str],
    bands: List[str],
    include_dataset_name: bool,
    product: str,
    wait=None,  # pylint: disable=unused-argument
) -> bool:
    """
    Processes satellite data for a given year and day of the year using the specified
    number of workers and datetime.

    Args:
        year (int): The year of the data to be processed.
        day_of_year (int): The day of the year for which to process the data.
        num_workers (int): The number of workers to use for parallel processing.
        dt (datetime.datetime): The datetime object representing the date to process.

    This function logs the processing activity, processes satellite data for specified
    products using `process_satellite`, and then builds a dataframe with `build_dataframe`.
    """
    # log(f"Start processing {product} satellite data...")
    file_paths = [file_paths] if not isinstance(file_paths, list) else file_paths

    for i in range(len(file_paths)):
        process_file(
            file_path=file_paths[i],
            bands=bands,  # aqui eu tirei o [i]
            include_dataset_name=include_dataset_name,
            product=product,
        )
    # log(f"End processing {product} satellite data...")
    return True


@task
def concat_processed_satellite_task(
    path,
    product,
    wait=None,  # pylint: disable=unused-argument
) -> bool:
    return concat_processed_satellite(
        path=path,
        product=product,
    )


@task
def build_dataframe_task(
    datetimes,
    verbose=False,
    overwrite=False,
    product="ABI-L2-RRQPEF",
    num_workers=1,
    location="rio_de_janeiro",
    timestep=10,
    value="RRQPE",
    band="RRQPE",
    wait=None,  # pylint: disable=unused-argument
):
    """
    Build dataframe
    """
    log("Start build dataframe...")
    build_dataframe_from_sat(
        datetimes=datetimes,
        verbose=verbose,
        overwrite=overwrite,
        product=product,
        num_workers=num_workers,
        location=location,
        timestep=timestep,
        value=value,
        band=band,
    )
    log("End build dataframe...")
    return True


@task
def get_predictions(
    dataframe_key,
    num_workers,
    cuda,
    wait=None,  # pylint: disable=unused-argument
) -> List:
    """
    get predictions
    """
    log("Start predictions...")
    return predict(dataframe_key=dataframe_key, num_workers=num_workers, cuda=cuda)


@task(nout=2)
def create_images(
    data_source: str = "SAT",
    num_workers: int = 10,
    nlags: int = 18,
    wait=None,  # pylint: disable=unused-argument
) -> List[str]:
    """
    Generate one image for each PNG file using a 3-hour forecast window (NLAGS=18).

    This function processes datasets to generate images for precipitation modeling, leveraging
    ground truth and model predictions for a specified timeframe. It supports datasets like
    satellite ("SAT") and radar ("MDN") and uses dataset-specific configurations for input files
    and grid definitions.

    Key Parameters and Behavior:
    - nlags: Number of time steps ahead (18 steps correspond to a 3-hour forecast window).
    - Data source (str): Specifies the dataset to use. Options are:
            - "SAT": For satellite data.
            - "MDN": For Mendanha radar data.
    - Configuration: Reads dataset-specific configurations from a JSON file (e.g., model names and
    plotting specs).
    - Files: Processes HDF5 ground truth and prediction files along with NumPy grid files to
    extract spatial and temporal data.
    - Parallelization: Uses multiprocessing to optimize the generation of forecast images for
    multiple time steps.

    Workflow:
    1. Load the dataset-specific ground truth file and grid definitions.
    2. Retrieve metadata like feature type, timestep interval, and last observed timestamp.
    3. Load model predictions based on the configuration file and skip models marked as non-plotting.
    4. Perform parallelized image generation for all forecast lags, ensuring efficient use of available resources.

    Example Usage:
        nlags = 18
        data_source = "SAT"  # or "MDN"
        generate_forecast_images(data_source)

    Notes:
    - Ground truth and grid file paths are defined in `dataset_dict`.
    - Model configurations are specified in JSON files (e.g., `real_time_config_{dataset}.json`).
    - Logs and warnings are generated for missing files or models.

    Returns
    -------
    List with all image's path
    """
    nlags = 18
    dataset_dict = {
        "SAT": {
            "ground_truth_file": "pipelines/precipitation_model/impa/data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-rio_de_janeiro/test.hdf",
            "grid_file": "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy",
        },
        "MDN": {
            "ground_truth_file": "pipelines/precipitation_model/impa/data/dataframes/MDN-d2CMAX-DBZH-real_time/test.hdf",
            "grid_file": "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=700m-256x256.npy",
        },
    }

    config = Path(
        f"pipelines/precipitation_model/impa/src/eval/real_time_config_{data_source}.json"
    )
    with open(config, "r") as json_file:
        specs_dict = json.load(json_file)

    ground_truth_df = h5py.File(dataset_dict[data_source]["ground_truth_file"])
    latlons = np.load(dataset_dict[data_source]["grid_file"])
    feature = ground_truth_df["what"].attrs["feature"]
    timestep = int(ground_truth_df["what"].attrs["timestep"])

    keys = get_dataset_keys(ground_truth_df)
    last_obs = keys[-1]
    last_obs_dt = pd.to_datetime(last_obs)

    log(f"last_obs: {last_obs}, last_obs_dt: {last_obs_dt}")

    # preds = [ground_truth_df]
    preds = [dataset_dict[data_source]["ground_truth_file"]]
    model_names = ["Ground truth"]
    for model_name in specs_dict["models"].keys():
        predictions = (
            f"pipelines/precipitation_model/impa/predictions_{data_source}/{model_name}.hdf"
        )
        if (
            "plot" in specs_dict["models"][model_name].keys()
            and specs_dict["models"][model_name]["plot"] is False
        ):
            continue
        try:
            # pred_hdf = h5py.File(predictions)
            preds.append(predictions)
            # preds.append(pred_hdf)
            model_names.append(model_name)
            log(f"preds: {preds}, \nmodel_names: {model_names}")
        except FileNotFoundError:
            print_warning(f"File {predictions} not found. Skipping model {model_name}...")
            continue

    # Criar a função parcialmente aplicada
    task_lag_img_partial = partial(
        task_lag_img,
        last_obs=last_obs,
        last_obs_dt=last_obs_dt,
        timestep=timestep,
        data_source=data_source,
        model_names=model_names,
        preds=preds,
        latlons=latlons,
        feature=feature,
    )

    with Pool(min(nlags, num_workers)) as pool:
        filenames = list(
            tqdm.tqdm(pool.imap(task_lag_img_partial, range(nlags, nlags + 1)), total=nlags)
        )
    log(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>\n {filenames}")
    return [[i] for i in filenames[0]], model_names


@task
def get_storage_destination_impa(path: str, model_version: int, model_names: List) -> str:
    """
    Get storage blob destinationa and the name of the source file
    """
    model_names = [i.replace(" ", "_").lower() for i in model_names]
    return [path + f"{i}/v{model_version}/3h/without_background" for i in model_names]
