# -*- coding: utf-8 -*-
# pylint: disable=import-error, invalid-name, missing-function-docstring
"""
Tasks
"""

import datetime

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from prefect import task
from prefeitura_rio.pipelines_utils.logging import log
from src.data.process.build_dataframe import build_dataframe
from src.data.process.process_satellite import process_satellite
from src.eval.predict_real_time import predict

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
def download_files_from_s3(relevant_dts, days_of_year, years):
    # Initialize the S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    hours = range(24)
    for i in range(4):
        day_of_year = days_of_year[i]
        year = years[i]
        print(f"Downloading the latest data for {relevant_dts[i].strftime('%Y-%m-%d')}...")
        for hour in hours:
            download_file_from_s3(s3, "ABI-L2-RRQPEF", year, day_of_year, hour)
            download_file_from_s3(s3, "ABI-L2-ACHAF", year, day_of_year, hour)


@task
def process_data(year, day_of_year, num_workers, dt):
    # process data
    log("Processing satellite data...")
    process_satellite(year=year, day=day_of_year, num_workers=num_workers, product="ABI-L2-RRQPEF")
    process_satellite(year=year, day=day_of_year, num_workers=num_workers, product="ABI-L2-ACHAF")
    build_dataframe(overwrite=True, num_workers=num_workers, dt=dt)


@task
def get_predictions(num_workers, cuda):
    """
    get predictions
    """
    predict(num_workers=num_workers, cuda=cuda)
