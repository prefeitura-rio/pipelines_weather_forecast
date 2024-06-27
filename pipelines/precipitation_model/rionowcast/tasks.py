# -*- coding: utf-8 -*-
"""
Tasks
"""
import os
import time

from pathlib import Path
from typing import Dict, Tuple  # , List

#  from typing import Callable, Dict, List, Tuple, Union

import basedosdados as bd
import pendulum
from prefect import task

from prefeitura_rio.pipelines_utils.infisical import get_secret
from prefeitura_rio.pipelines_utils.logging import log
from pipelines.constants import constants
from pipelines.precipitation_model.rionowcast.utils import bq_project, GypscieApi

@task()
def access_api():
    """
    Acess api and return it to be used in other requests
    """
    infisical_username = constants.INFISICAL_USERNAME.value
    infisical_password = constants.INFISICAL_PASSWORD.value
    username = get_secret(infisical_username)[infisical_username]
    password = get_secret(infisical_password)[infisical_password]
    log("\n\n[DEBUG]: username from infisical: {username} {type(username)} ")
    log("\n\n[DEBUG]: password from infisical: {password}")
    # info = json.loads(base64.b64decode(secret))
    # secret_name = f"DISCORD_WEBHOOK_URL_{monitor_slug.upper()}"
    # webhook_url = get_secret(secret_name=secret_name, environment=environment).get(secret_name)
    api = GypscieApi(username=username, password=password)

    return api


@task()
def get_stations_or_historical_data(
    dataset_info: dict,
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
    log(f"Downloading data and saving on {savepath}")
    bd.download(savepath=savepath, query=query, billing_project_id=bq_project())

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
def get_dataset_processor_info(api, processor_name: str) -> Tuple(dict, int):
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
) -> Dict:
    """
    Requisição de execução de um DatasetProcessor
    """
    log("\nStarting executing dataset processing")
    log(
        "processor_id",
        processor_id,
        "dataset_id",
        dataset_id,
        "environment_id",
        environment_id,
        "project_id",
        project_id,
    )

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

    log(task_response.status_code)
    log(task_response.json())
    log("\nFinish executing dataset processing")
    return task_response.json()


@task()
def wait_task_run(api, task_id) -> Dict:
    """
    Force flow wait for the end of data processing
    """
    if "task_id" in task_id.keys():
        _id = task_id.get("task_id")

        # Requisição do resultado da task_id
        response = api.get(
            path="status_processor_run/" + _id,
        )

    log(f"Response state: {response['state']}")
    while response["state"] == "STARTED":
        log("Transformation started")
        time.sleep(5)
        response = wait_task_run(api, task_id)

    if response["state"] != "SUCCESS":
        log("Error processing this dataset. Stop flow or restart this task")
    else:
        return response
