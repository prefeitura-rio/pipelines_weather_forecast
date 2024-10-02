# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download meteorological data, treat then, integrate and predict
"""

from prefect import case, Parameter
from prefect.run_configs import KubernetesRun
from prefect.storage import GCS

# from google.api_core.exceptions import Forbidden
from prefeitura_rio.pipelines_utils.custom import Flow  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_inject_bd_credentials,
)
from prefeitura_rio.pipelines_utils.logging import log
from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401
    create_table_and_upload_to_gcs,
    get_now_datetime,
    task_run_dbt_model_task,
)

from pipelines.constants import constants  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.rionowcast.schedules import (  # pylint: disable=E0611, E0401
    prediction_schedule,
    # update_schedule,
)

# from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
    # access_api,
    # calculate_start_and_end_date,
    # create_image,
    # query_data_from_gcp,
    # register_dataset_on_gypscie,
    # task_wait_run,

from pipelines.tasks import (  # pylint: disable=E0611, E0401
    task_create_partitions,
)

with Flow(
    name="WEATHER FORECAST: Previsão de Chuva - IMPA",
) as prediction_previsao_chuva_impa:

    #########################
    #  Define parameters    #
    #########################

    # Model parameters
    hours_from_past = Parameter("hours_from_past", required=True, default=6)
    start_date = Parameter("start_date", default=None, required=False)
    end_date = Parameter("end_date", default=None, required=False)

    # Gypscie parameters
    environment_id = Parameter("environment_id", default=1, required=False)
    domain_id = Parameter("domain_id", default=1, required=False)
    project_id = Parameter("project_id", default=1, required=False)
    workflow_id = Parameter("workflow_id", default=36, required=False)
    pre_processing_function_id = Parameter("pre_processing_function_id", default=43, required=False)
    load_data_function_id = Parameter("load_data_function_id", default=42, required=False)
    post_processing_function_id = Parameter(
        "post_processing_function_id", default=18, required=False
    )
    model_id = Parameter("model_id", default=18, required=False)
    # radar_data_id = Parameter("radar_data_id", default=178, required=False)
    # rain_gauge_data_id = Parameter("rain_gauge_data_id", default=179, required=False)
    grid_data_id = Parameter("grid_data_id", default=177, required=False)

    # Parameters for saving data on GCP
    materialize_after_dump = Parameter("materialize_after_dump", default=False, required=False)
    dump_mode = Parameter("dump_mode", default=False, required=False)
    dataset_id = mode_redis = Parameter("dataset_id", default="clima_rionowcast", required=False)
    table_id = Parameter("table_id", default="predicao_precipitacao", required=False)()


    #########################
    #  Start flow           #
    #########################

    with case(start_date, None):
        start_date, end_date = calculate_start_and_end_date(hours_from_past)

    
    image_path = create_image(geolocalized_prediction_datasets)
    # Save prediction on file
    prediction_data_path = task_create_partitions(
        geolocalized_prediction_datasets,
        partition_date_column="data_predicao",  # TODO: change column name
        # partition_columns=["ano_particao", "mes_particao", "data_particao"],
        savepath="model_prediction",
        suffix=now_datetime,
    )

    ##############################
    #  Save predictions on GCP   #
    ##############################

    # Upload data to BigQuery
    create_table = create_table_and_upload_to_gcs(
        data_path=prediction_data_path,
        dataset_id=dataset_id,
        table_id=table_id,
        dump_mode=dump_mode,
        biglake_table=False,
    )

    # Trigger DBT flow run
    with case(materialize_after_dump, True):
        run_dbt = task_run_dbt_model_task(
            dataset_id=dataset_id,
            table_id=table_id,
            # mode=materialization_mode,
            # materialize_to_datario=materialize_to_datario,
        )
        run_dbt.set_upstream(create_table)


##############################
#  Flow run parameters       #
##############################

prediction_previsao_chuva_impa.state_handlers = [handler_inject_bd_credentials]
prediction_previsao_chuva_impa.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
prediction_previsao_chuva_impa.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[constants.WEATHER_FORECAST_AGENT_LABEL.value],
)
prediction_previsao_chuva_impa.schedule = prediction_schedule
