# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download sattelite goes 16 data, treat then and predict
"""

from prefect import Parameter  # pylint: disable=E0611, E0401
from prefect.executors import LocalDaskExecutor  # pylint: disable=E0611, E0401
from prefect.run_configs import KubernetesRun  # pylint: disable=E0611, E0401
from prefect.storage import GCS  # pylint: disable=E0611, E0401

# from google.api_core.exceptions import Forbidden
from prefeitura_rio.pipelines_utils.custom import Flow  # pylint: disable=E0611, E0401

# from prefeitura_rio.pipelines_utils.logging import log
# pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_initialize_sentry,
    handler_inject_bd_credentials,
)

from pipelines.constants import constants  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.impa.schedules import (  # pylint: disable=E0611, E0401
    prediction_schedule,
)

# from pipelines.precipitation_model.impa.src.eval.viz.plot-real_time import create_images
from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
    build_dataframe_task,
    concat_processed_satellite_task,
    download_files_from_s3_task,
    get_filenames_to_process_task,
    get_predictions,
    get_relevant_dates_informations_task,
    get_start_datetime_task,
    process_satellite_task,
)
from pipelines.tasks import (  # pylint: disable=E0611, E0401; upload_files_to_storage,
    download_files_from_storage,
    get_storage_destination,
    unzip_files,
)

# from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401
#     create_table_and_upload_to_gcs,
#     get_now_datetime,
#     task_run_dbt_model_task,
# )


# from pipelines.tasks import task_create_partitions  # pylint: disable=E0611, E0401

# from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
# access_api,
# calculate_start_and_end_date,
# create_image,
# query_data_from_gcp,
# register_dataset_on_gypscie,
# task_wait_run,


with Flow(
    name="WEATHER FORECAST: Previsão de Chuva - IMPA",
    state_handlers=[
        handler_initialize_sentry,
        handler_inject_bd_credentials,
    ],
    parallelism=10,
    skip_if_running=False,
) as prediction_previsao_chuva_impa:

    #########################
    #  Define parameters    #
    #########################

    # Model parameters
    start_datetime = Parameter(
        "start_datetime",
        default="2024-10-17 10:00:00",
        required=False,
        # description="Datetime in YYYY-MM-dd HH:mm:ss format, UTC timezone",
    )
    num_workers = Parameter(
        "num_workers",
        default=8,
        required=False,
        # description="Number of workers to use for parallel processing",
    )
    cuda = Parameter("cuda", default=False, required=False)  # description="UseCUDA for prediction"
    # memory = Parameter("memory", default="16Gi", required=False)

    # Parameters for saving data on GCP
    materialize_after_dump = Parameter("materialize_after_dump", default=False, required=False)
    dump_mode = Parameter("dump_mode", default=False, required=False)
    dataset_id = Parameter("dataset_id", default="clima_previsao_chuva", required=False)
    table_id = Parameter("table_id", default="modelo_satelite_goes_16_impa", required=False)
    n_historical_hours = Parameter("n_historical_hours", default=6, required=False)

    download_base_path = "pipelines/precipitation_model/impa/data/raw/satellite"
    download_models_files = "files_SAT.zip"

    #########################
    #  Start flow           #
    #########################

    downloaded_ = download_files_from_storage(
        project="datario",
        bucket_name="datario-public",
        source_folder="cor-clima-imagens/predicao_precipitacao/impa/models_files/",
        destination_file_names=[download_models_files],
    )
    unziped_files = unzip_files(
        compressed_files=download_models_files,
        destination_folder="./",
        wait=downloaded_,
    )

    dt = get_start_datetime_task(start_datetime=start_datetime)
    relevant_dts, relevant_times = get_relevant_dates_informations_task(
        dt=dt,
        n_historical_hours=n_historical_hours,
    )

    # Download data from s3 from last 6h for RRQPE
    downloaded_files_rr = download_files_from_s3_task(
        product="ABI-L2-RRQPEF",
        relevant_times=relevant_times[:6],
        download_base_path=download_base_path,
    )
    # Download data from s3 from more hours for ACHAF because we need a dalay in this compared
    # to RRQPE to calculate parallax correction
    downloaded_files_achaf = download_files_from_s3_task(
        product="ABI-L2-ACHAF",
        relevant_times=relevant_times,
        download_base_path=download_base_path,
    )

    files_rr, bands_rr, include_dataset_name_rr = get_filenames_to_process_task(
        product="ABI-L2-RRQPEF",
        download_base_path=download_base_path,
        wait=downloaded_files_rr,
    )
    files_achaf, bands_achaf, include_dataset_name_achaf = get_filenames_to_process_task(
        product="ABI-L2-ACHAF",
        download_base_path=download_base_path,
        wait=downloaded_files_achaf,
    )

    data_processed_rr = process_satellite_task(
        file_paths=files_rr,
        bands=bands_rr,
        include_dataset_name=include_dataset_name_rr,
        product="ABI-L2-RRQPEF",
        wait=unziped_files,
    )
    data_processed_achaf = process_satellite_task(
        file_paths=files_achaf,
        bands=bands_achaf,
        include_dataset_name=include_dataset_name_achaf,
        product="ABI-L2-ACHAF",
        wait=unziped_files,
    )

    data_concat_rr = concat_processed_satellite_task(
        path="pipelines/precipitation_model/impa/data/processed_temp/satellite/ABI-L2-RRQPEF",
        product="ABI-L2-RRQPEF",
        wait=data_processed_rr,
    )
    data_concat_achaf = concat_processed_satellite_task(
        path="pipelines/precipitation_model/impa/data/processed_temp/satellite/ABI-L2-ACHAF",
        product="ABI-L2-ACHAF",
        wait=data_processed_achaf,
    )

    dfr = build_dataframe_task(
        datetimes=relevant_dts[:-12],
        overwrite=True,
        num_workers=num_workers,
        wait=[data_concat_rr, data_concat_achaf],
    )
    output_predict_filepaths = get_predictions(
        dataframe_key="SAT",
        num_workers=num_workers,
        cuda=cuda,
        wait=dfr,
    )

    destination_folder_models = get_storage_destination(
        path="cor-clima-imagens/previsao_chuva/impa/modelos"
    )

    # prediction_images_path = create_images()
    # destination_folder_images = get_storage_destination(
    #     path="cor-clima-imagens/previsao_chuva/impa/"
    # )
    # upload_files_to_storage(
    #     project="datario",
    #     bucket_name="datario-public",
    #     destination_folder=destination_folder_images,
    #     source_file_names=prediction_images_path,
    # )

    # image_path = create_image(geolocalized_prediction_datasets)
    # # Save prediction on file
    # prediction_data_path = task_create_partitions(
    #     geolocalized_prediction_datasets,
    #     partition_date_column="data_predicao",  # TODO: change column name
    #     # partition_columns=["ano_particao", "mes_particao", "data_particao"],
    #     savepath="model_prediction",
    #     suffix=now_datetime,
    # )

    # ##############################
    # #  Save predictions on GCP   #
    # ##############################

    # # Upload data to BigQuery
    # create_table = create_table_and_upload_to_gcs(
    #     data_path=output_predict_filepaths,
    #     dataset_id=dataset_id,
    #     table_id=table_id,
    #     dump_mode=dump_mode,
    #     biglake_table=False,
    # )

    # # Trigger DBT flow run
    # with case(materialize_after_dump, True):
    #     run_dbt = task_run_dbt_model_task(
    #         dataset_id=dataset_id,
    #         table_id=table_id,
    #         # mode=materialization_mode,
    #         # materialize_to_datario=materialize_to_datario,
    #     )
    #     run_dbt.set_upstream(create_table)


##############################
#  Flow run parameters       #
##############################

prediction_previsao_chuva_impa.state_handlers = [handler_inject_bd_credentials]
prediction_previsao_chuva_impa.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
prediction_previsao_chuva_impa.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[
        constants.WEATHER_FORECAST_AGENT_LABEL.value,
    ],
    cpu_request="500m",
    memory_limit="30Gi",
    memory_request="15Gi",
)
prediction_previsao_chuva_impa.schedule = prediction_schedule
prediction_previsao_chuva_impa.executor = LocalDaskExecutor(num_workers=50)
