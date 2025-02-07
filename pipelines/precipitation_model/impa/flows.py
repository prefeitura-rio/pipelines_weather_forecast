# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download sattelite goes 16 data, treat then and predict
"""

from prefect import Parameter, unmapped  # pylint: disable=E0611, E0401
from prefect.executors import LocalDaskExecutor  # pylint: disable=E0611, E0401
from prefect.run_configs import KubernetesRun  # pylint: disable=E0611, E0401
from prefect.storage import GCS  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.custom import Flow  # pylint: disable=E0611, E0401

# pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_initialize_sentry,
    handler_inject_bd_credentials,
)

from pipelines.constants import constants  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.impa.schedules import (  # pylint: disable=E0611, E0401
    prediction_schedule,
)
from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
    build_dataframe_task,
    concat_processed_satellite_task,
    create_images,
    download_files_from_s3_task,
    get_filenames_to_process_task,
    get_predictions,
    get_relevant_dates_informations_task,
    get_start_datetime_task,
    get_storage_destination_impa,
    process_satellite_task,
)
from pipelines.tasks import (  # pylint: disable=E0611, E0401; upload_files_to_storage,; task_create_partitions,
    download_files_from_storage,
    get_cpu_usage,
    get_disk_usage,
    get_memory_usage,
    remove_paths,
    unzip_files,
    upload_files_to_storage,
)

# from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401
#     create_table_and_upload_to_gcs,
#     get_now_datetime,
#     task_run_dbt_model_task,
# )


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
        default=None,
        required=False,
        # description="Datetime in YYYY-MM-dd HH:mm:ss format, UTC timezone",
    )
    num_workers = Parameter(
        "num_workers",
        default=4,
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
    model_version = Parameter("model_version", default=1, required=False)()

    download_base_path = "pipelines/precipitation_model/impa/data/raw/satellite"
    data_source = "SAT"
    download_models_files = f"files_{data_source}.zip"

    #########################
    #  Start flow           #
    #########################

    downloaded_ = download_files_from_storage(
        project="datario",
        bucket_name="datario-public",
        source_folder="cor-clima-imagens/predicao_precipitacao/impa/models_files",
        destination_file_names=[download_models_files],
    )
    unziped_files = unzip_files(
        compressed_files=[download_models_files],
        destination_folder="pipelines/precipitation_model/impa",
        wait=downloaded_,
    )

    dt = get_start_datetime_task(start_datetime=start_datetime)
    relevant_dts, relevant_times = get_relevant_dates_informations_task(
        dt=dt,
        n_historical_hours=n_historical_hours,
    )

    # Download data from s3 from last 6h for RRQPE
    downloaded_files_rr = download_files_from_s3_task.map(
        product=unmapped("ABI-L2-RRQPEF"),
        relevant_times=relevant_times[:8],
        download_base_path=unmapped(download_base_path),
    )
    # Download data from s3 from more hours for ACHAF because we need a dalay in this compared
    # to RRQPE to calculate parallax correction
    downloaded_files_achaf = download_files_from_s3_task.map(
        product=unmapped("ABI-L2-ACHAF"),
        relevant_times=relevant_times,
        download_base_path=unmapped(download_base_path),
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

    data_processed_rr = process_satellite_task.map(
        file_paths=files_rr,
        bands=unmapped(bands_rr),
        include_dataset_name=unmapped(include_dataset_name_rr),
        product=unmapped("ABI-L2-RRQPEF"),
    )
    data_processed_achaf = process_satellite_task.map(
        file_paths=files_achaf,
        bands=unmapped(bands_achaf),
        include_dataset_name=unmapped(include_dataset_name_achaf),
        product=unmapped("ABI-L2-ACHAF"),
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
        # wait=[data_concat_rr, data_concat_achaf],
        wait=[data_concat_rr, data_concat_achaf, unziped_files],
    )
    removed_paths = remove_paths(
        paths=[
            "pipelines/precipitation_model/impa/data/processed_temp",
            "pipelines/precipitation_model/impa/data/raw",
            download_models_files,
        ],
        wait=dfr,
    )
    cpu_usage = get_cpu_usage(wait=removed_paths)
    disk_usage = get_disk_usage(wait=removed_paths)
    memory_usage = get_memory_usage(wait=removed_paths)
    output_predict_filepaths = get_predictions(
        dataframe_key=data_source,
        num_workers=num_workers,
        cuda=cuda,
        wait=[dfr, removed_paths, disk_usage, memory_usage, cpu_usage],
    )

    prediction_images_path, model_names = create_images(
        data_source=data_source,
        num_workers=num_workers,
        nlags=18,
        wait=output_predict_filepaths,
    )
    destination_folder_images_ = get_storage_destination_impa(
        path="cor-clima-imagens/predicao_precipitacao/impa/",
        model_version=model_version,
        model_names=model_names,
    )
    upload_files_to_storage.map(
        project=unmapped("datario"),
        bucket_name=unmapped("datario-public"),
        destination_folder=destination_folder_images_,
        source_file_names=prediction_images_path,
    )

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
    cpu_limit="4",
    cpu_request="4",
    memory_limit="60Gi",
    memory_request="25Gi",
)
prediction_previsao_chuva_impa.schedule = prediction_schedule
prediction_previsao_chuva_impa.executor = LocalDaskExecutor(num_workers=1)
