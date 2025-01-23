# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download sattelite goes 16 data, treat then and predict
"""

# from pipelines.precipitation_model.impa.src.eval.viz.plot_real_time import create_images
from pipelines.precipitation_model.impa.utils import (  # pylint: disable=E0611, E0401
    concat_processed_satellite,
    download_file_from_s3,
    get_filenames_to_process,
    get_relevant_dates_informations,
    get_start_datetime,
)
from pipelines.precipitation_model.impa.src.data.process.build_dataframe_from_sat import (
    build_dataframe_from_sat,
)
from pipelines.precipitation_model.impa.src.data.process.process_satellite import (
    process_file,
)
from pipelines.precipitation_model.impa.src.eval.predict_real_time import predict
from pipelines.tasks import (  # pylint: disable=E0611, E0401
    get_storage_destination,
    upload_files_to_storage,
)


#########################
#  Define parameters    #
#########################

# Model parameters
start_datetime = "2024-10-17 10:00:00"
num_workers = 8

# Parameters for saving data on GCP
materialize_after_dump = False
dump_mode = "dev"
dataset_id = "clima_previsao_chuva"
table_id = "modelo_satelite_goes_16_impa"
n_historical_hours = 6

download_base_path = "pipelines/precipitation_model/impa/data/raw/satellite"
#########################
#  Start flow           #
#########################


get_predictions = predict
log=print

def process_satellite(
    file_paths,
    bands,
    include_dataset_name,
    product,
    wait=None,  # pylint: disable=unused-argument
) -> bool:
    log(f"Processing {product} satellite data...")
    for i in range(len(file_paths)):
        process_file(
            file_path=file_paths[i],
            bands=bands,
            include_dataset_name=include_dataset_name,
            product=product,
        )
    log(f"End processing {product} satellite data...")
    return True

def download_files_from_s3(
    product,
    relevant_times,
    download_base_path: str = "pipelines/precipitation_model/impa/data/raw/satellite",
):
    for relevant_time in relevant_times:
        download_file_from_s3(product, *relevant_time, download_base_path)
    return True

dt = get_start_datetime(start_datetime=start_datetime)
relevant_dts, relevant_times = get_relevant_dates_informations(
    dt=dt,
    n_historical_hours=n_historical_hours,
)
log(f"relevant_dts: {relevant_dts}")
log(f"relevant_dts[:-6]: {relevant_dts[:-6]}")
log(f"relevant_times: {relevant_times}")
log(f"relevant_times[:-6]: {relevant_times[:-6]}")
log(f"relevant_times[:7]: {relevant_times[:7]}")
# Download data from s3 from last 6h for RRQPE
downloaded_files_rr = download_files_from_s3(
    product="ABI-L2-RRQPEF",
    relevant_times=relevant_times[:6],
    download_base_path=download_base_path,
)
# Download data from s3 from more hours for ACHAF because we need a dalay in this compared
# to RRQPE to calculate parallax correction
downloaded_files_achaf = download_files_from_s3(
    product="ABI-L2-ACHAF",
    relevant_times=relevant_times,
    download_base_path=download_base_path,
)

files_rr, bands_rr, include_dataset_name_rr = get_filenames_to_process(
    product="ABI-L2-RRQPEF",
    download_base_path=download_base_path,
)
files_achaf, bands_achaf, include_dataset_name_achaf = get_filenames_to_process(
    product="ABI-L2-ACHAF",
    download_base_path=download_base_path,
)

data_processed_rr = process_satellite(
    file_paths=files_rr,
    bands=bands_rr,
    include_dataset_name=include_dataset_name_rr,
    product="ABI-L2-RRQPEF",
)
data_processed_achaf = process_satellite(
    file_paths=files_achaf,
    bands=bands_achaf,
    include_dataset_name=include_dataset_name_achaf,
    product="ABI-L2-ACHAF",
)

data_concat_rr = concat_processed_satellite(
    path="pipelines/precipitation_model/impa/data/processed_temp/satellite/ABI-L2-RRQPEF",
    product="ABI-L2-RRQPEF",
)
data_concat_achaf = concat_processed_satellite(
    path="pipelines/precipitation_model/impa/data/processed_temp/satellite/ABI-L2-ACHAF",
    product="ABI-L2-ACHAF",
)

dfr = build_dataframe_from_sat(
    datetimes=relevant_dts[:-12],
    overwrite=True,
    num_workers=num_workers,
)
output_predict_filepaths = get_predictions(dataframe_key="SAT", num_workers=num_workers)

destination_folder_models = get_storage_destination(
    path="cor-clima-imagens/previsao_chuva/impa/modelos"
)
print("destination_folder_models", destination_folder_models)
# upload_files_to_storage(
#     project="datario",
#     bucket_name="datario-public",
#     destination_folder=destination_folder_models,
#     source_file_names=output_predict_filepaths,
# )
