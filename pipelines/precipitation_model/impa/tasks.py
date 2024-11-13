# -*- coding: utf-8 -*-
# pylint: disable=import-error, invalid-name, missing-function-docstring
"""
Tasks
"""

import datetime

from prefect import task  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401

from pipelines.precipitation_model.impa.src.data.process.build_dataframe import (
    build_dataframe,
)
from pipelines.precipitation_model.impa.src.data.process.process_satellite import (
    process_satellite,
)
from pipelines.precipitation_model.impa.src.eval.predict_real_time import predict
from pipelines.precipitation_model.impa.utils import download_file_from_s3


@task
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


@task(nout=3)
def get_relevant_dates_informations(dt):
    """
    Get relevant dates iformations for the last 4 days
    """

    relevant_dts = [dt - datetime.timedelta(days=day_delta) for day_delta in range(4)]
    days_of_year = [dt.timetuple().tm_yday for dt in relevant_dts]
    years = [dt.year for dt in relevant_dts]
    return relevant_dts, days_of_year, years


@task
def download_files_from_s3(
    product, relevant_dts, days_of_year, years, download_base_path: str = "data/raw/satellite"
):
    """
    Download satellite data from AWS S3 bucket.

    Parameters
    ----------

    Returns
    -------
    None
    """
    hours = range(24)
    for i in range(4):
        day_of_year = days_of_year[i]
        year = years[i]
        print(f"Downloading the latest data for {relevant_dts[i].strftime('%Y-%m-%d')}...")
        for hour in hours:
            download_file_from_s3(product, year, day_of_year, hour, download_base_path)


@task
def process_satellite_task(
    year,
    day_of_year,
    num_workers,
    product,
    download_base_path: str = "data/raw/satellite",
    wait=None,
):
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
    process_satellite(
        year=year,
        day=day_of_year,
        num_workers=num_workers,
        product=product,
        download_base_path=download_base_path,
    )
    log(f"End processing {product} satellite data...")
    return True


@task
def build_dataframe_task(num_workers, dt, wait=None):
    """ """
    log("Start build dataframe...")
    build_dataframe(overwrite=True, num_workers=num_workers, dt=dt)
    log("End build dataframe...")
    return True


@task
def get_predictions(num_workers, cuda, wait=None):
    """
    get predictions
    """
    log("Start predictions...")
    return predict(num_workers=num_workers, cuda=cuda)
