# -*- coding: utf-8 -*-
# pylint: disable=import-error, invalid-name, missing-function-docstring, too-many-arguments
"""
Tasks
"""

from typing import List, Tuple

from prefect import task  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401

from pipelines.precipitation_model.impa.src.data.process.build_dataframe_from_sat import (
    build_dataframe_from_sat,
)
from pipelines.precipitation_model.impa.src.data.process.process_satellite import (
    process_file,
)
from pipelines.precipitation_model.impa.src.eval.predict_real_time import predict
from pipelines.precipitation_model.impa.utils import (
    concat_processed_satellite,
    download_file_from_s3,
    get_filenames_to_process,
    get_relevant_dates_informations,
    get_start_datetime,
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
    relevant_times,
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
    file_paths,
    bands,
    include_dataset_name,
    product,
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
    log(f"Processing {product} satellite data...")
    for i in range(len(file_paths)):
        process_file(
            file_path=file_paths[i],
            bands=bands[i],
            include_dataset_name=include_dataset_name,
            product=product,
        )
    log(f"End processing {product} satellite data...")
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
