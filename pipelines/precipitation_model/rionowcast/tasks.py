# -*- coding: utf-8 -*-
"""
Tasks
"""
import os
from time import sleep
from pathlib import Path
from typing import Dict, List  # Tuple

# import basedosdados as bd
# from basedosdados.download.base import google_client
from basedosdados.upload.base import Base
from google.cloud import bigquery
import pandas as pd
import pendulum
from prefect import task
from prefect.engine.signals import ENDRUN
from prefect.engine.state import Failed

from prefeitura_rio.pipelines_utils.infisical import get_secret
from prefeitura_rio.pipelines_utils.logging import log
from pipelines.constants import constants
from pipelines.precipitation_model.rionowcast.utils import GypscieApi, wait_task_run


# noqa E302, E303
@task()
def access_api():
    """# noqa E303
    Acess api and return it to be used in other requests
    """
    infisical_username = constants.INFISICAL_USERNAME.value
    infisical_password = constants.INFISICAL_PASSWORD.value

    # username = get_secret(secret_name="USERNAME", path="/gypscie", environment="prod")
    # password = get_secret(secret_name="PASSWORD", path="/gypscie", environment="prod")

    username = get_secret(infisical_username, path="/gypscie")[infisical_username]
    password = get_secret(infisical_password, path="/gypscie")[infisical_password]
    api = GypscieApi(username=username, password=password)

    return api


@task()
def get_billing_project_id(
    bd_project_mode: str = "prod",
    billing_project_id: str = None,
) -> str:
    """
    Get billing project id
    """
    if not billing_project_id:
        log("Billing project ID was not provided, trying to get it from environment variable")
    try:
        bd_base = Base()
        billing_project_id = bd_base.config["gcloud-projects"][bd_project_mode]["name"]
        log(f"Billing project ID was inferred from environment variables: {billing_project_id}")
    except KeyError:
        pass
    if not billing_project_id:
        raise ValueError(
            "billing_project_id must be either provided or inferred from environment variables"
        )
    log(f"Billing project ID: {billing_project_id}")
    return billing_project_id


def download_data_from_bigquery(query: str, billing_project_id: str) -> pd.DataFrame:
    """ADD"""
    # pylint: disable=E1124, protected-access
    # client = google_client(billing_project_id, from_file=True, reauth=False)
    # job_config = bigquery.QueryJobConfig()
    # # job_config.dry_run = True

    # # Get data
    # log("Querying data from BigQuery")
    # job = client["bigquery"].query(query, job_config=job_config)
    # https://github.com/prefeitura-rio/pipelines_rj_iplanrio/blob/ecd21c727b6f99346ef84575608e560e5825dd38/pipelines/painel_obras/dump_data/tasks.py#L39
    bq_client = bigquery.Client(
        credentials=Base(bucket_name="rj-cor")._load_credentials(mode="prod"),
        project=billing_project_id,
    )
    job = bq_client.query(query)
    while not job.done():
        sleep(1)

    # Get data
    # log("Querying data from BigQuery")
    # job = client["bigquery"].query(query)
    # while not job.done():
    #     sleep(1)
    log("Getting result from query")
    results = job.result()
    log("Converting result to pandas dataframe")
    dfr = results.to_dataframe()
    log("End download data from bigquery")
    return dfr


@task()
def get_stations_or_historical_data(
    dataset_info: dict,
    billing_project_id: str,
    data_type: str = "historical",
    start_date: str = None,
    end_date: str = None,
) -> Path:
    """
    Download data from stations or historical data changing param data_type.
    data_type: str = "historical" or "station,
    """
    log(f"Start downloading {dataset_info['table_id']} {data_type} Data")

    directory_path = Path("data/input/")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    actual_timestamp = pendulum.now("America/Sao_Paulo").format("YYYYMMDDhhmmss")
    savepath = directory_path / f"{dataset_info['filename']}_{actual_timestamp}.csv"

    # pylint: disable=consider-using-f-string
    query = """
        SELECT
            *
        FROM rj-cor.{}.{}
        """.format(
        dataset_info["dataset_id"],
        dataset_info["table_id"],
    )

    # pylint: disable=consider-using-f-string
    if data_type == "historical":
        filter_query = """
            WHERE data_particao BETWEEN '{}' AND '{}'
        """.format(
            start_date, end_date
        )
        query += filter_query

    log(f"Query to be downloaded:\n{query}")

    dfr = download_data_from_bigquery(query=query, billing_project_id=billing_project_id)
    log(f"Saving data on {savepath}")
    dfr.to_csv(savepath, index=False)
    # bd.download(savepath=savepath, query=query, billing_project_id=billing_project_id)

    log(f"{dataset_info['table_id']} {type} data saved on {savepath}")
    return savepath


@task()
def register_dataset(api, filepath: Path, domain_id: int = 1) -> Dict:
    """
    Register dataset on gypscie and return its informations like id
    """
    log(f"\nStart registring dataset by sending {filepath} Data to Gypscie")

    data = {
        "domain_id": domain_id,
        "name": str(filepath).split("/")[-1].split(".csv")[0],  # pylint: disable=use-maxsplit-arg
    }
    log(type(data), data)
    files = {
        "files": open(file=filepath, mode="rb"),  # pylint: disable=consider-using-with
    }

    response = api.post(path="datasets", data=data, files=files)

    log(response)
    log(response.json())
    return response.json()


@task(nout=2)
def get_dataset_processor_info(api, processor_name: str):
    """
    Geting dataset processor information
    """
    log(f"Getting dataset processor info for {processor_name}")
    dataset_processors_response = api.get(
        path="dataset_processors",
    )

    # log(dataset_processors_response)
    dataset_processor_id = None
    for response in dataset_processors_response:
        if response.get("name") == processor_name:
            dataset_processor_id = response["id"]
            # log(response)
            # log(response["id"])
    return dataset_processors_response, dataset_processor_id

    # if not dataset_processor_id:
    #     log(f"{processor_name} not found. Try adding it.")


@task()
# pylint: disable=too-many-arguments
def execute_dataset_processor(
    api,
    processor_id: int,
    dataset_id: list,  # como pegar os vários datasets
    environment_id: int,
    project_id: int,
    parameters: dict
    # adicionar campos do dataset_processor
) -> List:
    """
    Requisição de execução de um DatasetProcessor
    """
    log("\nStarting executing dataset processing")

    task_response = api.post(
        path="processor_run",
        json_data={
            "dataset_id": dataset_id,
            "environment_id": environment_id,
            "parameters": parameters,
            "processor_id": processor_id,
            "project_id": project_id,
        },
    )

    response = wait_task_run(api, task_response.json())

    if response["state"] != "SUCCESS":
        failed_message = "Error processing this dataset. Stop flow or restart this task"
        log(failed_message)
        task_state = Failed(failed_message)
        raise ENDRUN(state=task_state)

    output_datasets = response["result"]["output_datasets"]  # returns a list with datasets
    log(f"\nFinish executing dataset processing, we have {len(output_datasets)} datasets")
    return output_datasets


@task()
def predict(api, model_id: int, dataset_id: int, project_id: int) -> dict:
    """
    Requisição de execução de um processo de Predição
    """
    print("Starting prediction")
    response = api.post(
        path="predict",
        data={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "project_id": project_id,
        },
    )
    print(f"Prediction ended. Response: {response}, {response.json()}")
    return response.json()
