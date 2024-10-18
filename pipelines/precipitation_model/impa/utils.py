# -*- coding: utf-8 -*-
# pylint: disable=C0103, C0302
"""
Utils
"""
import pathlib

from prefeitura_rio.pipelines_utils.logging import log


def download_file_from_s3(s3, product, year, day_of_year, hour):
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
    BUCKET_NAME = "noaa-goes16"
    # create parent folders
    prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
    parent_folder = pathlib.Path(f"data/raw/satellite/{prefix}")
    parent_folder.mkdir(parents=True, exist_ok=True)

    # download files
    log(f"Bucket name = {BUCKET_NAME} and prefix = {prefix}")
    s3_result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/")
    for obj in s3_result.get("Contents", []):
        key = obj["Key"]
        file_name = key.split("/")[-1].split(".")[0]
        filepath = pathlib.Path(f"data/raw/satellite/{prefix}/{file_name}.nc")
        if filepath.exists():
            continue
        s3.download_file(BUCKET_NAME, key, filepath)
        log(f"Downloaded {key}")
