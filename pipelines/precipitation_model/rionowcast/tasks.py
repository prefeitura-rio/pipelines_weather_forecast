# -*- coding: utf-8 -*-
# pylint: disable= C0207
"""
Tasks
"""
import datetime
import os
from pathlib import Path
from time import sleep
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pendulum  # pylint: disable=E0611, E0401
from basedosdados import Base  # pylint: disable=E0611, E0401
from google.cloud import bigquery  # pylint: disable=E0611, E0401
from prefect import task  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401

from pipelines.utils.utils_wf import convert_dtypes

# @task()
# def get_billing_project_id(
#     bd_project_mode: str = "prod",
#     billing_project_id: str = None,
# ) -> str:
#     """
#     Get billing project id.
#     OBS: not workin in this basedosdados version
#     """
#     if not billing_project_id:
#         log("Billing project ID was not provided, trying to get it from environment variable")
#     try:
#         bd_base = Base()
#         billing_project_id = bd_base.config["gcloud-projects"][bd_project_mode]["name"]
#         log(f"Billing project ID was inferred from environment variables: {billing_project_id}")
#     except KeyError:
#         pass
#     if not billing_project_id:
#         raise ValueError(
#             "billing_project_id must be either provided or inferred from environment variables"
#         )
#     log(f"Billing project ID: {billing_project_id}")
#     return billing_project_id


def download_data_from_bigquery(query: str, billing_project_id: str) -> pd.DataFrame:
    """ADD"""
    # pylint: disable=E1124, protected-access
    # client = google_client(billing_project_id, from_file=True, reauth=False)
    # job_config = bigquery.QueryJobConfig()
    # # job_config.dry_run = True

    # # Get data
    log("Querying data from BigQuery")
    # job = client["bigquery"].query(query, job_config=job_config)
    # https://github.com/prefeitura-rio/pipelines_rj_iplanrio/blob/ecd21c727b6f99346ef84575608e560e5825dd38/pipelines/painel_obras/dump_data/tasks.py#L39

    bq_client = bigquery.Client(
        credentials=Base(bucket_name="rj-cor")._load_credentials(mode="prod"),
        project=billing_project_id,
    )
    job = bq_client.query(query)
    while not job.done():
        sleep(1)
    log("Getting result from query")
    results = job.result()
    log("Converting result to pandas dataframe")
    dfr = results.to_dataframe()
    log("End download data from bigquery")

    # Get data
    # log("Querying data from BigQuery")
    # job = client["bigquery"].query(query)
    # while not job.done():
    #     sleep(1)
    return dfr


@task(nout=3)
def calculate_start_and_end_date(
    hours_from_past: int = 6,
    end_historical_datetime: str = None,
) -> tuple[str, str]:
    """
    Calculate the start and end datetime based on a specified number of hours in the past.

    The function computes two datetime values: the start and end times.
    The `end_historical_datetime_date` serves as the reference endpoint for calculations, and
    the `hours_from_past` parameter determines how far back in time the start date will be.
    Both values are rounded to the nearest full hour, and the result is returned as strings in
    the format 'yyyy-mm-dd hh:mm:ss' (UTC).

    Parameters:
    ----------
    hours_from_past : int
        Number of hours to subtract from `end_historical_datetime_date` to determine the start
        datetime.
    end_historical_datetime_date : str, optional
        The ending datetime as a string in the format 'yyyy-mm-dd hh:mm:ss' in UTC timezone.
        If not provided, the current UTC time (`datetime.datetime.utcnow()`) is used as the default.

    Returns:
    -------
    tuple[str, str]
        A tuple containing two strings:
        - The start datetime (earlier timestamp).
        - The end datetime (reference timestamp).

    Raises:
    ------
    ValueError
        If `end_historical_datetime_date` is provided but does not follow the expected format.

    Example:
    -------
    >>> calculate_start_and_end_date(6, "2024-11-13 12:00:00")
    ('2024-11-13 06:00:00', '2024-11-13 12:00:00')
    """
    hours_from_past = int(hours_from_past)
    if not end_historical_datetime:
        end_historical_datetime = datetime.datetime.utcnow()
    else:
        try:
            end_historical_datetime = datetime.datetime.strptime(
                end_historical_datetime, "%Y-%m-%d %H:%M:%S"
            )
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

    end_datetime = pendulum.instance(end_historical_datetime).replace(
        minute=0, second=0, microsecond=0
    )
    start_datetime = end_datetime.subtract(hours=hours_from_past)

    # Ajustar para o fuso horário de Brasília
    end_datetime_brasilia = end_datetime.in_timezone("America/Sao_Paulo")

    print(f"Start datetime in UTC: {start_datetime}, End datetime in UTC: {end_datetime}")

    return (
        start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime_brasilia.strftime("%Y-%m-%d %H:%M:%S"),
    )


@task()
def query_data_from_gcp(  # pylint: disable=too-many-arguments
    dataset_id: str,
    table_id: str,
    billing_project_id: str,
    start_datetime: str = None,
    end_datetime: str = None,
    filename: str = "data",
    save_format: str = "csv",
) -> Path:
    """
    Download historical data from source.
    format: csv or parquet
    """
    log(f"Start downloading {dataset_id}.{table_id} data")

    directory_path = Path(f"{dataset_id}_{table_id}")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    savepath = directory_path / f"{filename}.{save_format}"
    start_date, end_date = start_datetime[:10], end_datetime[:10]
    # pylint: disable=consider-using-f-string
    # noqa E262
    query = """
        SELECT
            * EXCEPT (update_time, model_version, ano_particao, mes_particao, data_particao)
        FROM rj-cor.{}.{}
        WHERE data_particao BETWEEN '{}' AND '{}'
        AND datetime >= '{}' AND datetime < '{}'
        """.format(
        dataset_id, table_id, start_date, end_date, start_datetime, end_datetime
    )

    log(f"Query used to download data:\n{query}")

    dfr = download_data_from_bigquery(query=query, billing_project_id=billing_project_id)
    dtype_mapping = {
        "station_id": "int64",
        "datetime": "datetime64[ns, UTC]",
        "precipitation": "float64",
        "latitude": "float64",
        "longitude": "float64",
        "altitude": "int64",
        "horizontal_reflectivity_mean": "float64",
    }

    dfr = convert_dtypes(dfr, dtype_mapping)
    if save_format == "csv":
        dfr.to_csv(savepath, index=False)
    elif save_format == "parquet":
        dfr.to_parquet(savepath, index=False)
    # bd.download(savepath=savepath, query=query, billing_project_id=billing_project_id)

    log(f"{table_id} data saved on {savepath}")
    return savepath


@task
def create_image(data, filename) -> List:
    """
    Create image using Geolocalized data or the numpy array from desnormalized_data function
    Exemplo de código que usei pra gerar uma imagem vindo de um xarray:

    def create_and_save_image(data: xr.xarray, variable) -> Path:
        plt.figure(figsize=(10, 10))

        # Use the Geostationary projection in cartopy
        axis = plt.axes(projection=ccrs.PlateCarree())

        lat_max, lon_max = (
            -21.708288842894145,
            -42.36573106186053,
        )  # canto superior direito
        lat_min, lon_min = (
            -23.793855217170343,
            -45.04488171189226,
        )  # canto inferior esquerdo

        extent = [lon_min, lat_min, lon_max, lat_max]
        img_extent = [extent[0], extent[2], extent[1], extent[3]]

        # Define the color scale based on the channel
        colormap = "jet"  # White to black for IR channels

        # Plot the image
        img = axis.imshow(data, origin="upper", extent=img_extent, cmap=colormap, alpha=0.8)

        # Add coastlines, borders and gridlines
        axis.coastlines(resolution='10m', color='black', linewidth=0.8)
        axis.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)


        grdln = axis.gridlines(
            crs=ccrs.PlateCarree(),
            color="gray",
            alpha=0.7,
            linestyle="--",
            linewidth=0.7,
            xlocs=np.arange(-180, 180, 1),
            ylocs=np.arange(-90, 90, 1),
            draw_labels=True,
        )
        grdln.top_labels = False
        grdln.right_labels = False

        plt.colorbar(
            img,
            label=variable.upper(),
            extend="both",
            orientation="horizontal",
            pad=0.05,
            fraction=0.05,
        )

        output_image_path = Path(os.getcwd()) / "output" / "images"

        save_image_path = output_image_path / (f"{variable}.png")

        if not output_image_path.exists():
            output_image_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_image_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.show()
        return save_image_path
    """
    save_images_path = []
    images = data[0][0, 0]
    for i in range(3):
        plt.imshow(images[i], cmap="viridis")
        plt.axis("off")

        base_path = f"{os.getcwd()}/{i + 1}h"
        os.makedirs(base_path, exist_ok=True)
        save_filename = f"{base_path}/{filename}.png"
        # save_filename = f"{base_path}/{filename}_{i + 1}h.png"
        # não pode ter nada no final do nome, para ter no começo tem que
        # alterar a busca no bucket na api
        plt.savefig(save_filename, bbox_inches="tight")
        save_images_path.append(save_filename)
        log(f"Imagem {i + 1} salva como {save_filename}")
        plt.close()
    log(f"Images saved on {save_images_path}")
    log(os.listdir("./"))
    return save_images_path
