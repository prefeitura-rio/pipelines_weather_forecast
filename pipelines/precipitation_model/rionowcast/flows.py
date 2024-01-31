# -*- coding: utf-8 -*-
from tasks import ReadParquetTask, StoreDataInLakeTask, DataframeToArrowTask, DataIntegratorTask
from prefect import Flow, Parameter
from prefect.run_configs import KubernetesRun
from prefect.storage import GCS

from pipelines.constants import constants

# Initialize your logger and data lake handler
logger = Logger.get_logger()
data_lake_handler = DataLakeWrapper()

# Instantiate tasks
read_parquet_task = ReadParquetTask(data_lake_handler)
store_data_in_lake_task = StoreDataInLakeTask(data_lake_handler)
dataframe_to_arrow_task = DataframeToArrowTask()
data_integrator_task = DataIntegratorTask(logger, data_lake_handler)

# Define the flow
with Flow("WeatherDataIntegrationFlow") as flow:
    # Define parameters if needed
    sources = Parameter("sources", default=["all"])
    period = Parameter("period", default=["start_date", "end_date"])
    non_shared_feature_handler = Parameter("non_shared_feature_handler", default=None)

    # Connect tasks based on your logic
    parquet_result = read_parquet_task(remote_path="...", columns=["..."])
    arrow_table = dataframe_to_arrow_task(df=parquet_result)
    
    # More connections based on your existing logic

    # Connect DataIntegratorTask
    data_integrator_result = data_integrator_task(
        non_shared_feature_handler=non_shared_feature_handler,
        sources=sources,
        period=period
    )

with Flow(
    name="weather_forecast: Nome do objetivo - Descrição detalhada do objetivo",
) as exemplo__nome_do_objetivo__greet_flow:
    # Parameters
    name = Parameter("name", default="weather_forecast")

    # Tasks
    greet_task = greet(name)


# Storage and run configs
exemplo__nome_do_objetivo__greet_flow.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
exemplo__nome_do_objetivo__greet_flow.run_config = KubernetesRun(image=constants.DOCKER_IMAGE.value)
