# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download meteorological data, treat then, integrate and predict
"""

from prefect import Parameter
from prefect.run_configs import KubernetesRun
from prefect.storage import GCS

from prefeitura_rio.pipelines_utils.custom import Flow
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_inject_bd_credentials,
)

from pipelines.constants import constants
from pipelines.precipitation_model.rionowcast.schedules import (
    update_schedule,
)

# from pathlib import Path

from pipelines.precipitation_model.rionowcast.tasks import (
    access_api,
    get_billing_project_id,
    get_dataset_processor_info,
    get_stations_or_historical_data,
    execute_dataset_processor,
    register_dataset,
    wait_task_run,
)

with Flow(
    name="WEATHER FORECAST: Previsão de Chuva - Rionowcast",
    # code_owners=[
    #     "paty",
    # ],
) as wf_previsao_chuva_rionowcast:

    # Data parameters
    # start_date = Parameter("start_date", default=None, required=False)
    # end_date = Parameter("end_date", default=None, required=False)
    # weather_dateset_info = Parameter(
    #     "weather_dataset_info",
    #     default={
    #         "dataset_id": "clima_estacao_meteorologica",
    #         "table_id": "meteorologia_inmet",
    #         "filename": "weather_station_bq"
    #     },
    #     required=False
    # )
    # pluviometer_dataset_info = Parameter(
    #     "pluviometer_dataset_info",
    #     default={
    #         "dataset_id": "clima_pluviometer",
    #         "table_id": "taxa_precipitacao_alertario",
    #         "filename": "gauge_station_bq",
    #     },
    #     required=False
    # )
    bd_project_mode = Parameter("bd_project_mode", default="prod", required=False)
    billing_project_id = Parameter("billing_project_id", default="rj-cor", required=False)

    start_date, end_date = "2024-02-02", "2024-02-03"

    weather_dataset_info = {
        "dataset_id": "clima_estacao_meteorologica",
        "table_id": "meteorologia_inmet",
        "filename": "weather_station_bq",
    }

    pluviometer_dataset_info = {
        "dataset_id": "clima_pluviometer",
        "table_id": "taxa_precipitacao_alertario",
        "filename": "gauge_station_bq",
    }

    # Gypscie parameters
    project_name = "rionowcast_precipitation"
    processor_name = Parameter("processor_name", default="etl_alertario21", required=True)
    environment_id = 1
    domain_id = 1
    project_id = 1

    api = access_api()

    # Get processor information on gypscie
    dataset_processor_response, dataset_processor_id = get_dataset_processor_info(
        api, processor_name
    )

    billing_project_id = get_billing_project_id(bd_project_mode)
    # # Download pluviometric and meteorological data
    # if weather_dataset_info:
    #     print("Downloading Meteorological Data")
    #     meteorological_path = get_stations_or_historical_data(
    #       weather_dataset_info, "historical", start_date, end_date)
    #     print(f"Meteorological Data saved on {meteorological_path}")
    #     meteorological_dataset_response = register_dataset(api, meteorological_path, domain_id)

    #     meteorological_processor_parameters = {
    #                     "dataset1": str(pluviometrical_path).split("/")[-1],
    #                     "station_type": "rain_gauge",
    #                     }
    #     # # Send data to be processed and treated
    #     task_id = execute_dataset_processor(
    #         api,
    #         processor_id=dataset_processor_id,
    #         dataset_id=[meteorological_dataset_response["id"]],
    #         environment_id=environment_id,
    #         project_id=project_id,
    #         parameters=meteorological_processor_parameters,
    #     )

    #     task_response = wait_task_run(api, task_id)
    #     task_response

    if pluviometer_dataset_info:
        pluviometrical_path = get_stations_or_historical_data(
            pluviometer_dataset_info, billing_project_id, "historical", start_date, end_date
        )
        pluviometrical_path.set_upstream(api)
        # pluviometrical_path = Path('data/input/rain_gauge_station_20240625111229.csv')
        print(f"Pluviometer Data saved on {pluviometrical_path}")
        pluviometrical_dataset_response = register_dataset(api, pluviometrical_path, domain_id)

        pluviometrical_processor_parameters = {
            "dataset1": str(pluviometrical_path).rsplit("/", maxsplit=1)[-1],
            "station_type": "rain_gauge",
        }
        # Send data to be processed and treated
        task_id = execute_dataset_processor(
            api,
            processor_id=dataset_processor_id,
            dataset_id=[pluviometrical_dataset_response["id"]],
            environment_id=environment_id,
            project_id=project_id,
            parameters=pluviometrical_processor_parameters,
        )

        task_response = wait_task_run(api, task_id)

wf_previsao_chuva_rionowcast.state_handlers = [handler_inject_bd_credentials]
wf_previsao_chuva_rionowcast.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
wf_previsao_chuva_rionowcast.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[constants.WEATHER_FORECAST_AGENT_LABEL.value],
)
wf_previsao_chuva_rionowcast.schedule = update_schedule

# https://github.com/prefeitura-rio/pipelines_rj_escritorio/blob/2433238db27adb1213059832f238495b9ecb5043/pipelines/deteccao_alagamento_cameras/flooding_detection/flows.py#L112
# https://linen.prefect.io/t/13543083/how-do-i-run-the-same-subflow-concurrently-for-items-in-a-li
