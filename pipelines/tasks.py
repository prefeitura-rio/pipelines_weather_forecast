# -*- coding: utf-8 -*-
# pylint: disable=R0914,W0613,W0102,R0913
"""
Common  Tasks for rj-cor
"""

import gzip
import json
import os
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import List, Union

import basedosdados as bd  # pylint: disable=E0611, E0401
import pandas as pd
import pendulum  # pylint: disable=E0611, E0401
import psutil
from google.cloud import storage  # pylint: disable=E0611, E0401
from prefect import task  # pylint: disable=E0611, E0401
from prefect.triggers import all_successful  # pylint: disable=E0611, E0401
from prefeitura_rio.core import settings  # pylint: disable=E0611, E0401

# pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.infisical import get_secret
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.pandas import (  # pylint: disable=E0611, E0401
    dump_header_to_file,
    parse_date_columns,
    to_partitions,
)
from prefeitura_rio.pipelines_utils.redis_pal import (  # pylint: disable=E0611, E0401
    get_redis_client,
)

from pipelines.utils.utils_wf import (
    treat_redis_output,  # get_redis_client_from_infisical,
)

# from redis_pal import RedisPal


@task(checkpoint=False)
def task_get_redis_client(
    infisical_host_env: str = "REDIS_HOST",
    infisical_port_env: str = "REDIS_PORT",
    infisical_db_env: str = "REDIS_DB",
    infisical_password_env: str = "REDIS_PASSWORD",
    infisical_secrets_path: str = "/",
):
    """
    Gets a Redis client.

    Args:
        infisical_host_env: The environment variable for the Redis host.
        infisical_port_env: The environment variable for the Redis port.
        infisical_db_env: The environment variable for the Redis database.
        infisical_password_env: The environment variable for the Redis password.

    Returns:
        The Redis client.
    """
    redis_host = get_secret(infisical_host_env, path=infisical_secrets_path)[infisical_host_env]
    redis_port = int(
        get_secret(infisical_port_env, path=infisical_secrets_path)[infisical_port_env]
    )
    redis_db = int(get_secret(infisical_db_env, path=infisical_secrets_path)[infisical_db_env])
    redis_password = get_secret(infisical_password_env, path=infisical_secrets_path)[
        infisical_password_env
    ]
    return get_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
    )


@task
def task_build_redis_hash(dataset_id: str, table_id: str, name: str = None, mode: str = "prod"):
    """
    Helper function for building a key to redis
    """
    redis_hash = dataset_id + "." + table_id
    if name:
        redis_hash = redis_hash + "." + name
    if mode == "dev":
        redis_hash = f"{mode}.{redis_hash}"
    return redis_hash


# @task
# def get_on_redis(
#     dataset_id: str,
#     table_id: str,
#     mode: str = "prod",
#     wait=None,
# ) -> list:
#     """
#     Get filenames saved on Redis.
#     converti em get_redis_client_from_infisical
#     """
#     redis_client = get_redis_client()
#     key = build_redis_key(dataset_id, table_id, "files", mode)
#     files_on_redis = redis_client.get(key)
#     files_on_redis = [] if files_on_redis is None else files_on_redis
#     files_on_redis = list(set(files_on_redis))
#     files_on_redis.sort()
#     return files_on_redis


@task
def task_get_redis_output(
    redis_client,
    redis_hash: str = None,
    redis_key: str = None,
    treat_output: bool = True,
    expected_output_type: str = "list",
    is_df: bool = False,
):
    """
    Get Redis output. Use get to obtain a df from redis or hgetall if is a key value pair.
    Redis output example: {b'date': b'2023-02-27 07:29:04'}
    expected_output_type "list"m "dict", "df"
    """

    if is_df or expected_output_type == "df":
        json_data = redis_client.get(redis_hash)
        log(f"[DEGUB] json_data {json_data}")
        if json_data:
            # If data is found, parse the JSON string back to a Python object (dictionary)
            data_dict = json.loads(json_data)
            # Convert the dictionary back to a DataFrame
            return pd.DataFrame(data_dict)

        return pd.DataFrame()

    if redis_hash and redis_key:
        output = redis_client.hget(redis_hash, redis_key)
    elif redis_key:
        output = redis_client.get(redis_key)
    else:
        output = redis_client.hgetall(redis_hash)

    if not output:
        output = [] if expected_output_type == "list" else {}

    log(f"Output from redis before treatment{type(output)}\n{output}")
    if len(output) > 0 and treat_output and not isinstance(output, list):
        output = treat_redis_output(output)
    log(f"Output from redis {type(output)}\n{output}")
    return output


# @task(trigger=all_successful)
# def save_on_redis(
#     dataset_id: str,
#     table_id: str,
#     mode: str = "prod",
#     files: list = [],
#     keep_last: int = 50,
#     wait=None,
# ) -> None:
#     """
#     Set the last updated time on Redis.
#     """
#     redis_client = get_redis_client_from_infisical()
#     key = build_redis_key(dataset_id, table_id, "files", mode)
#     files = list(set(files))
#     print(">>>> save on redis files ", files)
#     files.sort()
#     files = files[-keep_last:]
#     redis_client.set(key, files)


@task(trigger=all_successful)
def task_save_on_redis(
    redis_client,
    values,
    redis_hash: str = None,
    redis_key: str = None,
    keep_last: int = 50,
    wait=None,
) -> None:
    """
    Save values on redis. If values are a list, order ir and keep only last names.
    """

    if isinstance(values, list):
        values = list(set(values))
        values.sort()
        values = values[-keep_last:]

    if isinstance(values, dict):
        values = json.dumps(values)

    log(f"Saving files {values} on redis {redis_hash} {redis_key}")

    if redis_hash and redis_key:
        redis_client.hset(redis_hash, redis_key, values)
    elif redis_key:
        redis_client.set(redis_key, values)
    log(f"Saved to Redis hash: {redis_hash}, key: {redis_key}, value: {values}")


# @task(nout=2)
# def get_storage_destination(filename: str, path: str) -> Tuple[str, str]:
#     """
#     Get storage blob destinationa and the name of the source file
#     """
#     destination_blob_name = f"cor-clima-imagens/radar/mendanha/{filename}.png"
#     source_file_name = f"{path}/{filename}.png"
#     log(f"File destination_blob_name {destination_blob_name}")
#     log(f"File source_file_name {source_file_name}")
#     return destination_blob_name, source_file_name
@task
def get_storage_destination(path: str, filename: str = None) -> str:
    """
    Get storage blob destinationa and the name of the source file
    """
    destination_blob_name = f"{path}/{filename}" if filename else path
    log(f"File destination_blob_name {destination_blob_name}")
    return destination_blob_name


@task
def get_storage_destination_model(path: str, filename: str = None) -> str:
    """
    Get storage blob destinationa and the name of the source file
    """
    destination_blob_name = f"{path}/{filename}" if filename else path
    log(f"File destination_blob_name {destination_blob_name}")
    return destination_blob_name


@task
def upload_files_to_storage(
    project: str, bucket_name: str, destination_folder: str, source_file_names: List[str]
) -> None:
    """
    Upload multiple files to GCS, where the destination folder is the directory in the bucket
    and source_file_names is a list of file paths to upload.

    project="datario"
    bucket_name="datario-public"
    destination_folder="cor-clima-imagens/radar/mendanha/"
    source_file_names=["/local/path/image1.png", "/local/path/image2.png"]
    """
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)

    log(f"Uploading {len(source_file_names)} files to {destination_folder}.")
    for file_path in source_file_names:
        file_name = file_path.split("/")[-1]
        blob = bucket.blob(f"{destination_folder}/{file_name}")
        blob.upload_from_filename(file_path)

        log(
            f"File {file_name} from {file_path} sent to {destination_folder} on bucket {bucket_name}."
        )


# def upload_files_to_storage(
#     project: str,
#     bucket_name: str,
#     destination_blob_name: str,
#     source_file_name: List[str],
# ) -> None:
#     """
#     Upload files to GCS

#     project="datario"
#     bucket_name="datario-public"
#     destination_blob_name=f"cor-clima-imagens/radar/mendanha/{filename}.png"
#     source_file_name=f"{path}/{filename}.png"
#     """
#     storage_client = storage.Client(project=project)
#     bucket = storage_client.bucket(bucket_name)
#     # Cria um blob (o arquivo dentro do bucket)
#     blob = bucket.blob(destination_blob_name)
#     for i in source_file_name:
#         blob.upload_from_filename(i)
#         log(f"File {i} sent to {destination_blob_name} on bucket {bucket_name}.")


@task
def download_files_from_storage(
    project: str,
    bucket_name: str,
    source_folder: str,
    destination_file_names: List[str],
    wait=None,  # pylint: disable=unused-argument
) -> None:
    """
    Download multiple files from GCS, where the source folder is the directory in the bucket
    and destination_file_names is a list of file paths to download.

    project="datario"
    bucket_name="datario-public"
    source_folder="cor-clima-imagens/radar/mendanha/"
    destination_file_names=["/local/path/image1.png", "/local/path/image2.png"]
    """
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)

    log(f"Downloading {len(destination_file_names)} files from {source_folder}.")
    for file_name in destination_file_names:
        blob = bucket.blob(f"{source_folder}/{file_name}")
        blob.download_to_filename(file_name)

        log(f"File {file_name} downloaded from {source_folder} on bucket {bucket_name}.")
    return "downloaded"


@task
def save_dataframe(
    dfr: pd.DataFrame,
    partition_column: str,
    suffix: str = "current_timestamp",
    path: str = "temp",
    wait=None,  # pylint: disable=unused-argument
) -> Union[str, Path]:
    """
    Salvar dfr tratados em csv para conseguir subir pro GCP
    """

    prepath = Path(path)
    prepath.mkdir(parents=True, exist_ok=True)

    partition_column = "data_medicao"
    dataframe, partitions = parse_date_columns(dfr, partition_column)
    if suffix == "current_timestamp":
        suffix = pendulum.now("America/Sao_Paulo").strftime("%Y%m%d%H%M")

    to_partitions(
        data=dataframe,
        partition_columns=partitions,
        savepath=prepath,
        data_type="csv",
        suffix=suffix,
    )
    log(f"Data saved on {prepath}")
    return prepath


@task
def task_create_partitions(
    data: Union[pd.DataFrame, str],
    partition_date_column: str,
    # partition_columns: List[str],
    savepath: str = "temp",
    data_type: str = "csv",
    suffix: str = None,
    build_json_dataframe: bool = False,
    dataframe_key_column: str = None,
) -> Path:  # sourcery skip: raise-specific-error
    """
    Create task for to_partitions
    """
    if isinstance(data, str):
        data = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
    data, partition_columns = parse_date_columns(data, partition_date_column)
    log(f"Created partition columns {partition_columns} and data first row now is {data.iloc[0]}")
    saved_files = to_partitions(
        data=data,
        partition_columns=partition_columns,
        savepath=savepath,
        data_type=data_type,
        suffix=suffix,
        build_json_dataframe=build_json_dataframe,
        dataframe_key_column=dataframe_key_column,
    )
    log(f"Partition saved files {saved_files}")
    log(f"Returned path {savepath}, {type(savepath)}")
    return Path(savepath)


@task
def convert_parameter_to_type(parameter_, new_type):
    """Function to convert model version from Parameter to type specified on new_type"""
    print(f"Actual type of parameter {parameter_}: {type(parameter_)}")
    print(f"New type {type(new_type(parameter_))}")
    return new_type(parameter_)


@task(
    max_retries=settings.TASK_MAX_RETRIES_DEFAULT,
    retry_delay=timedelta(seconds=settings.TASK_RETRY_DELAY_DEFAULT),
)
def create_table_and_upload_to_gcs(
    data_path: str | Path,
    dataset_id: str,
    table_id: str,
    dump_mode: str,
    bucket_name: str = None,
    biglake_table: bool = True,
) -> None:
    """
    Create table using BD+ and upload to GCS.
    """
    bd_version = bd.__version__
    log(f"USING BASEDOSDADOS {bd_version}")
    # pylint: disable=C0103
    tb = bd.Table(dataset_id=dataset_id, table_id=table_id)
    table_staging = f"{tb.table_full_name['staging']}"
    # pylint: disable=C0103
    log("DEGUB: before st")
    st = bd.Storage(dataset_id=dataset_id, table_id=table_id)
    bucket_name = bucket_name if bucket_name else st.bucket_name
    storage_path = f"{bucket_name}.staging.{dataset_id}.{table_id}"
    storage_path_link = (
        f"https://console.cloud.google.com/storage/browser/{bucket_name}"
        f"/staging/{dataset_id}/{table_id}"
    )
    log(f"Storage path link: {storage_path_link}")

    # prod datasets is public if the project is datario. staging are private im both projects
    dataset_is_public = tb.client["bigquery_prod"].project == "datario"

    #####################################
    #
    # MANAGEMENT OF TABLE CREATION
    #
    #####################################
    log("STARTING TABLE CREATION MANAGEMENT")
    log(f"GETTING DATA FROM: {data_path}")
    if dump_mode == "append":
        if tb.table_exists(mode="staging"):
            log(f"MODE APPEND: Table ALREADY EXISTS:" f"\n{table_staging}" f"\n{storage_path_link}")
        else:
            # the header is needed to create a table when dosen't exist
            log("MODE APPEND: Table DOSEN'T EXISTS\nStart to CREATE HEADER file")
            header_path = dump_header_to_file(data_path=data_path)
            log("MODE APPEND: Created HEADER file:\n" f"{header_path}")

            tb.create(
                path=header_path,
                if_storage_data_exists="replace",
                if_table_exists="replace",
                biglake_table=biglake_table,
                dataset_is_public=dataset_is_public,
            )

            log(
                "MODE APPEND: Sucessfully CREATED A NEW TABLE:\n"
                f"{table_staging}\n"
                f"{storage_path_link}"
            )  # pylint: disable=C0301

            st.delete_table(mode="staging", bucket_name=bucket_name, not_found_ok=True)
            log(
                "MODE APPEND: Sucessfully REMOVED HEADER DATA from Storage:\n"
                f"{storage_path}\n"
                f"{storage_path_link}"
            )  # pylint: disable=C0301
    elif dump_mode == "overwrite":
        if tb.table_exists(mode="staging"):
            log(
                "MODE OVERWRITE: Table ALREADY EXISTS, DELETING OLD DATA!\n"
                f"{storage_path}\n"
                f"{storage_path_link}"
            )  # pylint: disable=C0301
            st.delete_table(mode="staging", bucket_name=bucket_name, not_found_ok=True)
            log(
                "MODE OVERWRITE: Sucessfully DELETED OLD DATA from Storage:\n"
                f"{storage_path}\n"
                f"{storage_path_link}"
            )  # pylint: disable=C0301
            # delete only staging table and let DBT overwrite the prod table
            tb.delete(mode="staging")
            log(
                "MODE OVERWRITE: Sucessfully DELETED TABLE:\n" f"{table_staging}\n"
            )  # pylint: disable=C0301

        # the header is needed to create a table when dosen't exist
        # in overwrite mode the header is always created
        log("MODE OVERWRITE: Table DOSEN'T EXISTS\nStart to CREATE HEADER file")
        header_path = dump_header_to_file(data_path=data_path)
        log("MODE OVERWRITE: Created HEADER file:\n" f"{header_path}")

        tb.create(
            path=header_path,
            if_storage_data_exists="replace",
            if_table_exists="replace",
            biglake_table=biglake_table,
            dataset_is_public=dataset_is_public,
        )

        log(
            "MODE OVERWRITE: Sucessfully CREATED TABLE\n"
            f"{table_staging}\n"
            f"{storage_path_link}"
        )

        st.delete_table(mode="staging", bucket_name=bucket_name, not_found_ok=True)
        log(
            f"MODE OVERWRITE: Sucessfully REMOVED HEADER DATA from Storage\n:"
            f"{storage_path}\n"
            f"{storage_path_link}"
        )  # pylint: disable=C0301

    #####################################
    #
    # Uploads a bunch of files using BD+
    #
    #####################################

    log("STARTING UPLOAD TO GCS")
    if tb.table_exists(mode="staging"):
        # the name of the files need to be the same or the data doesn't get overwritten
        tb.append(filepath=data_path, if_exists="replace")

        log(
            f"STEP UPLOAD: Successfully uploaded {data_path} to Storage:\n"
            f"{storage_path}\n"
            f"{storage_path_link}"
        )
    else:
        # pylint: disable=C0301
        log("STEP UPLOAD: Table does not exist in STAGING, need to create first")

    return data_path


@task
def unzip_files(
    compressed_files: List[str],
    destination_folder: str = "./",
    wait=None,  # pylint: disable=unused-argument
) -> List[str]:
    """
    Unzip .zip and .gz files to destination folder.
    """
    log(f"Compressed files: {compressed_files} will be sent to {destination_folder}.")
    compressed_files = [
        zip_file if zip_file.endswith((".zip", ".gz")) else zip_file + ".zip"
        for zip_file in compressed_files
    ]
    os.makedirs(destination_folder, exist_ok=True)

    extracted_files = []
    for file in compressed_files:
        if file.endswith(".zip"):
            log("zip file found")
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(destination_folder)
                extracted_files.extend(
                    [os.path.join(destination_folder, f) for f in zip_ref.namelist()]
                )
        elif file.endswith(".gz"):
            output_file = os.path.join(destination_folder, os.path.basename(file)[:-3])
            with gzip.open(file, "rb") as gz_file:
                with open(output_file, "wb") as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            extracted_files.append(output_file)
    log(f"Extracted files: {extracted_files}")
    return extracted_files


@task
def remove_paths(
    paths: List[str],
    wait=None,  # pylint: disable=unused-argument
) -> bool:
    """
    If path is a file, remove it. If path is a directory, remove all files inside it.
    """
    for path in paths:
        path_obj = Path(path)

        if path_obj.is_file():
            try:
                path_obj.unlink()
                log(f"File removed: {path}")
            except Exception as e:
                log(f"Error removing file {path}: {e}")

        elif path_obj.is_dir():
            # Se for um diretório, remove os arquivos dentro dele
            for file in path_obj.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    log(f"Error removing file {file}: {e}")
            log(f"Files from {path} removed.")

        else:
            log(f"Path: {path} doesn't exist.")

    return True


@task
def get_disk_usage(path="/", wait=None):  # pylint: disable=unused-argument)
    """
    Retorna o uso atual do disco para o caminho especificado.
    Por padrão, verifica o uso no diretório raiz (/).
    """
    disk = psutil.disk_usage(path)
    total_gb = disk.total / 1e9
    used_gb = disk.used / 1e9
    free_gb = disk.free / 1e9
    percent_used = disk.percent
    disk_usage = {
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "percent_used": percent_used,
    }
    log(
        f"Espaço em disco: {disk_usage['used_gb']:.2f}GB usados \
            / {disk_usage['total_gb']:.2f}GB totais ({disk_usage['percent_used']}%)"
    )
    return disk_usage


@task
def get_memory_usage(wait=None):  # pylint: disable=unused-argument)
    """
    Retorna o uso atual de memória RAM em GB e a porcentagem utilizada.
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    used_gb = mem.used / 1e9
    percent_used = mem.percent
    mem_usage = {"total_gb": total_gb, "used_gb": used_gb, "percent_used": percent_used}
    log(
        f"Memória RAM: {mem_usage['used_gb']:.2f}GB \
            / {mem_usage['total_gb']:.2f}GB ({mem_usage['percent_used']}%)"
    )
    return mem_usage


@task
def get_cpu_usage(wait=None):  # pylint: disable=unused-argument)
    """
    Retorna o uso atual da CPU em núcleos utilizados e em porcentagem.
    """
    total_cores = psutil.cpu_count(logical=True)  # Número total de cores lógicos
    cpu_percent = psutil.cpu_percent(interval=1)  # Uso da CPU em %

    # Estimando quantos núcleos estão sendo usados (aproximadamente)
    used_cores = (cpu_percent / 100) * total_cores
    cpu_usage = {"used_cores": used_cores, "cpu_percent": cpu_percent}
    print(f"Uso da CPU: {cpu_usage['used_cores']:.2f} núcleos (~{cpu_usage['cpu_percent']}%)")
    return cpu_usage
