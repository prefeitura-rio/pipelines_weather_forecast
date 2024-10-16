# -*- coding: utf-8 -*-
import datetime
import pathlib
from argparse import ArgumentParser
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from joblib import Parallel, delayed
from pipelines.precipitation_model.impa.src.data.process.build_dataframe import build_dataframe
from pipelines.precipitation_model.impa.src.data.process.process_satellite import process_satellite
from pipelines.precipitation_model.impa.src.eval.predict_real_time import predict

# from itertools import product


BUCKET_NAME = "noaa-goes16"


def download_data(product, year, day_of_year, hour):
    # Initialize the S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    # create parent folders
    prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
    parent_folder = pathlib.Path(f"data/raw/satellite/{prefix}")
    parent_folder.mkdir(parents=True, exist_ok=True)

    # download files
    s3_result = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/")
    for obj in s3_result.get("Contents", []):
        key = obj["Key"]
        file_name = key.split("/")[-1].split(".")[0]
        filepath = pathlib.Path(f"data/raw/satellite/{prefix}/{file_name}.nc")
        if filepath.exists():
            continue
        s3.download_file(BUCKET_NAME, key, filepath)
        print(f"Downloaded {key}")


def download_parallel_multiprocessing(time_tuples):
    with ProcessPoolExecutor() as executor:
        future_to_key = {
            executor.submit(download_data, time_tuple): time_tuple for time_tuple in time_tuples
        }

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--datetime",
        type=str,
        default=None,
        help="Datetime in ISO format, UTC timezone",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for prediction")
    args = parser.parse_args()

    if args.datetime is None:
        dt = datetime.datetime.now(tz=datetime.timezone.utc)
    else:
        dt = datetime.datetime.fromisoformat(args.datetime)

    print(f"Running predictions on datetime [{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC]")

    relevant_dts = [dt - datetime.timedelta(days=day_delta) for day_delta in range(4)]
    days_of_year = [dt.timetuple().tm_yday for dt in relevant_dts]
    years = [dt.year for dt in relevant_dts]

    hours = range(24)

    # time_tuples = zip([s3]*4, ["ABI-L2-RRQPEF"]*4, years, days_of_year)
    # time_tuples = product(time_tuples, hours)
    # for key, result in download_parallel_multiprocessing(time_tuples):
    #     if result is not None:
    #         print(f"Error downloading {key}: {result}")

    for i in range(4):
        day_of_year = days_of_year[i]
        year = years[i]
        print(f"Downloading the latest data for {relevant_dts[i].strftime('%Y-%m-%d')}...")

        download_hour = partial(download_data, "ABI-L2-RRQPEF", year, day_of_year)
        Parallel(n_jobs=args.num_workers)(delayed(download_hour)(hour) for hour in hours)
        download_hour = partial(download_data, "ABI-L2-ACHAF", year, day_of_year)
        Parallel(n_jobs=args.num_workers)(delayed(download_hour)(hour) for hour in hours)

    # process data
    print("Processing satellite data...")
    process_satellite(
        year=years[0],
        day=days_of_year[0],
        num_workers=args.num_workers,
        product="ABI-L2-RRQPEF",
    )
    process_satellite(
        year=years[0],
        day=days_of_year[0],
        num_workers=args.num_workers,
        product="ABI-L2-ACHAF",
    )
    build_dataframe(overwrite=True, num_workers=args.num_workers, dt=dt)

    predict(num_workers=args.num_workers, cuda=args.cuda)
