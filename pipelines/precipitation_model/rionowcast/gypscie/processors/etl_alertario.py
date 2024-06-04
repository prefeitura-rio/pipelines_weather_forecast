# -*- coding: utf-8 -*-

"""
# subir arquivo que veio direto do alertario já com latitude e longitude das estações no gypscie
# subir esse arquivo em zip com o conda, scrip, mlproject e enviar no campo de dataset processor
# depois ver como fazer esse envio via api
# chamar api pra ver os resultados
# escritorio_dados/cor-alagamentos/lncc.txt
# $ python etl_alertario.py -t

"""

import argparse
import os
import sys
import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd

# import pyarrow as pa
# import pyarrow.parquet as pq
# from dotenv import load_dotenv
# from utils.logging import Logger
# from src.utils.datalake import DataLakeWrapper
from utils.meteorological import (
    cyclic_time_encoding,
    cyclic_month_encoding,
    # cyclic_wind_encoding,
)
from utils.filesystem import make_path, destroy_path

warnings.filterwarnings("ignore")


# # class DataLakeHandler:
# #     """Class responsible for collecting and storing processed data in the data lake
# #     """
# #     def __init__(self) -> None:
# #         self._dl = DataLakeWrapper()


# #     def read_parquet(self, remote_path, columns=None, filters=None):
# #         ddf = self._dl.read_parquet_from_lake(
# #             remote_path,
# #             columns=columns,
# #             filters=filters
# #         )
# #         return ddf


# #     def store_data(self, table:pa.Table, path):
# #         pq.write_to_dataset(
# #             table,
# #             root_path=f's3://{path}',
# #             partition_cols=['year', 'month'],
# #             filesystem=self._dl.get_filesystem(),
# #         )


class FileSystemHandler:
    """
    ADD
    """

    # logger = Logger.get_logger()

    @classmethod
    def create_path(cls, path):
        """
        ADD
        """
        try:
            make_path(path)
        except FileExistsError:
            # cls.logger.warning(f'Path "{path}" already exists. Ignoring create path command...')
            pass
        except:
            raise


# class HBVIndicatorFixer:

#     def __init__(self, local_data_path) -> pd.DataFrame:
#         self._local_data_path = local_data_path
#         # self._logger = Logger.get_logger()
#         # self._dlc = DataLakeWrapper().get_client()
#         self._df_hbv = self._load_hbv_calendar()

#     def fix(self, df: pd.DataFrame):
#         # self._logger.debug("Fixing HBV started ...")
#         if df["datetime"].isnull().sum():
#             raise Exception(
#                 "There are null values in datetime column. Fix it before running this HBVFixer"
#             )
#         year = df.iloc[0]["datetime"].year
#         hbv_start_date, hbv_end_date = self._get_hbv_dates(year)
#         df["fixed_hbv"] = None
#         if not pd.isnull(hbv_start_date):
#             df = self._fix_start_interval(df, hbv_start_date)
#         df = self._fix_non_hbv_interval(df, hbv_start_date, hbv_end_date)
#         if not pd.isnull(hbv_end_date):
#             df = self._fix_end_interval(df, hbv_end_date)
#         self._run_assertions(df)
#         df["HBV"] = df["fixed_hbv"].astype(bool)
#         df = df.drop(columns=["fixed_hbv"])
#         # self._logger.debug("Fixing HBV done.")
#         return df

#     def _load_hbv_calendar(self):
#         local_path = os.path.join(self._local_data_path, "bdst.parquet")
#         self._dlc.fget_object(
#             bucket_name="landing",
#             object_name="brazilian_daylight_saving_time/bdst.parquet",
#             file_path=local_path,
#         )
#         df = pd.read_parquet(local_path)
#         return df

#     def _get_hbv_dates(self, year):
#         try:
#             hbv_start_date = self._df_hbv[self._df_hbv["year"] == year]["started"].item()
#             hbv_end_date = self._df_hbv[self._df_hbv["year"] == year]["ended"].item()
#         except Exception:
#             raise Exception(f"HBV calendar not found for year {year}")
#         return hbv_start_date, hbv_end_date

#     def _fix_start_interval(self, df: pd.DataFrame, hbv_start_date):
#         # self._logger.debug("Checking HBV inconsistent interval")
#         print("Checking HBV inconsistent interval")
#         hbv_inconsistent_interval = (
#             hbv_start_date,
#             hbv_start_date + pd.Timedelta(minutes=59, seconds=59),
#         )
#         if df.query(
#             f'datetime >= "{hbv_inconsistent_interval[0]}" and datetime <= "{hbv_inconsistent_interval[1]}"'
#         ).shape[0]:
#             message = f"There are records in HBV inconsistent interval from {hbv_inconsistent_interval[0]} to {hbv_inconsistent_interval[1]}"
#             raise Exception(message)
#         # self._logger.debug("Setting records after HBV start datetime")
#         print("Setting records after HBV start datetime")
#         hbv_start_datetime = hbv_start_date + pd.Timedelta(hours=1)
#         df.loc[df["datetime"] >= hbv_start_datetime, "fixed_hbv"] = True
#         return df

#     def _fix_non_hbv_interval(self, df: pd.DataFrame, hbv_start_date, hbv_end_date):
#         if pd.isnull(hbv_start_date) and pd.isnull(hbv_end_date):
#             df.loc[:, "fixed_hbv"] = False
#             return df
#         c1 = df["datetime"] >= hbv_end_date
#         c2 = df["datetime"] < hbv_start_date
#         if (not pd.isnull(hbv_start_date)) and (not pd.isnull(hbv_end_date)):
#             df.loc[c1 & c2, "fixed_hbv"] = False
#         elif not pd.isnull(hbv_end_date):
#             df.loc[c1, "fixed_hbv"] = False
#         elif not pd.isnull(hbv_start_date):
#             df.loc[c2, "fixed_hbv"] = False
#         return df

#     def _fix_end_interval(self, df, hbv_end_date):
#         # self._logger.debug("Setting records before HBV ambiguous interval")
#         hbv_ambiguous_interval_start = hbv_end_date - pd.Timedelta(hours=1)
#         df.loc[df["datetime"] < hbv_ambiguous_interval_start, "fixed_hbv"] = True
#         # self._logger.debug("Setting records within HBV ambiguous interval")
#         hbv_ambiguous_mask = df["fixed_hbv"].isnull()
#         df_hbv_ambiguous = df.loc[hbv_ambiguous_mask]
#         min_hbv_ambiguous_datetime = df_hbv_ambiguous["datetime"].min()
#         max_hbv_ambiguous_datetime = df_hbv_ambiguous["datetime"].max()
#         if (
#             min_hbv_ambiguous_datetime < hbv_ambiguous_interval_start
#             or max_hbv_ambiguous_datetime > hbv_end_date
#         ):
#             raise Exception("There are records outside HBV ambiguous interval")
#         ambiguous_duplicated_index = df_hbv_ambiguous[
#             df_hbv_ambiguous.duplicated(subset=["station", "datetime"])
#         ].index
#         df.loc[ambiguous_duplicated_index, "fixed_hbv"] = False
#         df.loc[df["fixed_hbv"].isnull(), "fixed_hbv"] = True
#         return df

#     def _run_assertions(self, df: pd.DataFrame):
#         # self._logger.debug("Checking if all records are marked")
#         if df.query("fixed_hbv.isnull()").shape[0]:
#             message = "There are records without fixed HBV indicator"
#             raise Exception(message)
#         # self._logger.debug("Checking inconsistencies between original and processed HBV indicator")
#         c1 = (df["HBV"] == "HBV") & (df["fixed_hbv"] == False)
#         c2 = (df["HBV"] != "HBV") & (df["fixed_hbv"] == True)
#         numeric_cols = df.select_dtypes(include="number").columns
#         if df.loc[c1 | c2].dropna(subset=numeric_cols, how="all").shape[0]:
#             INCONSISTENCIES_PATH = os.path.join("data", "hbv_inconsistencies")
#             FileSystemHandler.create_path(INCONSISTENCIES_PATH)
#             year = df.iloc[0]["datetime"].year
#             instrument = "weather_station" if "wind_dir" in numeric_cols else "rain_gauge"
#             df.loc[c1 | c2].dropna(subset=numeric_cols, how="all").to_parquet(
#                 os.path.join("data", "hbv_inconsistencies", f"{instrument}_{year}.parquet")
#             )
#             # self._logger.warning(
#             #     f'There are inconsistencies between original and processed HBV indicator. Check "{INCONSISTENCIES_PATH}" folder'
#             # )
#             print(
#                 f'There are inconsistencies between original and processed HBV indicator. Check "{INCONSISTENCIES_PATH}" folder'
#             )


class DataOptimizer:
    """
    ADD
    """

    def __init__(self, local_data_path, station_type) -> None:
        # self._logger = Logger.get_logger()
        # self._dlc = DataLakeWrapper().get_client()
        # self._dlfs = DataLakeWrapper().get_filesystem()
        # self.remote_bucket = remote_bucket
        # self.remote_prefix = remote_prefix
        self.local_data_path = local_data_path
        # FileSystemHandler.create_path(self.local_data_path)
        # self._hbv_fixer = HBVIndicatorFixer(self.local_data_path)
        self.local_extracted_path = os.path.join(self.local_data_path, "extracted")
        # FileSystemHandler.create_path(self.local_extracted_path)
        self.columns = []
        self.numeric_cols = []
        self.instrument = None
        self.variables = {}
        self.station_type = station_type

    def process_objects(self):
        """
        ADD
        """
        try:
            dfr = pd.read_csv(f"{self.station_type}_bq.csv")
            # print(f"File opened\n{dfr.iloc[0]}")
            print(f"File opened\n{dfr.head()}")

            dfr = dfr.reset_index(drop=True)
            # code to stage data for HBV ambiguous fixer
            # station_type = self.remote_prefix.split('/')[0]
            # dfr.to_pickle(f'data/hbv_fixer_{station_type}_YYYY.pkl')
            print("Starting casting columns")
            dfr = self._cast_columns(dfr)
            print("Performing final steps...")
            dfr.reset_index(drop=True, inplace=True)
            dfr["year"] = dfr["datetime"].dt.year
            dfr["month"] = dfr["datetime"].dt.month
            attrs = {"naming_authority": "Alerta Rio", "timezone": "America/Sao_Paulo"}
            attrs.update(self.instrument)
            attrs.update(self.variables)

            save_path = f"{self.station_type}_optimized.csv"
            print(f"File before saving\n{dfr.iloc[0]}")
            print(f"Saving file on path {save_path}.")
            dfr[self.columns].to_csv(save_path, index=False)
        except Exception as error:
            print(f"Failed to process {self.station_type}: {error}")
        # destroy_path(self.local_extracted_path, recursive=True)
        # destroy_path(self.local_data_path, recursive=True)

    def _cast_columns(self, dfr: pd.DataFrame):
        """
        Casting columns to right format, create datetime column and fix HBV.
        """
        dfr["id_estacao"] = dfr["id_estacao"].astype("category")
        for col in self.numeric_cols:
            print(f"Casting {col} column...")
            dfr[col] = pd.to_numeric(dfr[col], errors="coerce")
        print("Building string datetime column...")
        dfr["datetime"] = dfr["data_particao"] + " " + dfr["horario"]
        print("Casting datetime column...")
        dfr["datetime"] = pd.to_datetime(
            dfr["datetime"], dayfirst=True, errors="coerce", format="%Y/%m/%d %H:%M:%S"
        )
        dfr.drop(["data_particao", "horario"], axis=1, inplace=True)
        # if not args.skip_hbv_fixer:
        #     print(f"Fixing HBV indicator...")
        #     dfr = self._hbv_fixer.fix(dfr)
        # print('Localizing datetime column to "America/Sao_Paulo" timezone...')
        # dfr["datetime"] = dfr["datetime"].dt.tz_localize("America/Sao_Paulo", ambiguous=dfr["HBV"])
        # dfr.drop("HBV", axis=1, inplace=True)
        return dfr


# # class WeatherStationDataOptimizer(DataOptimizer):

# #     def __init__(self, remote_bucket, remote_prefix, local_data_path) -> None:
# #         super().__init__(remote_bucket, remote_prefix, local_data_path)
# #         self.columns = [
# #             "data",
# #             "hora",
# #             "HBV",
# #             "precipitation",
# #             "wind_dir",
# #             "wind_speed",
# #             "temperature",
# #             "pressure",
# #             "humidity",
# #         ]
# #         self.numeric_cols = self.columns[3:]
# #         self.skiprows = 6
# #         self.instrument = {"instrument": "Weather Station"}
# #         self.variables = self._build_variables()

# #     def _build_variables(self):
# #         values = [
# #             "precipitation (mm/h)",
# #             "wind_dir (degrees)",
# #             "wind_speed (km/h)",
# #             "temperature (degrees Celsius)",
# #             "pressure (hPa)",
# #             "humidity (%)",
# #         ]
# #         keys = [f"variable_{i}" for i in range(1, len(values) + 1)]
# #         return dict(zip(keys, values))


class RainGaugeDataOptimizer(DataOptimizer):
    """
    ADD
    """

    def __init__(self, local_data_path, station_type) -> None:
        super().__init__(local_data_path, station_type)
        # self.columns = ["data", "hora", "HBV", "15min", "01h", "04h", "24h", "96h"]
        self.columns = [
            "id_estacao",
            "acumulado_chuva_15_min",
            "acumulado_chuva_1_h",
            "acumulado_chuva_4_h",
            "acumulado_chuva_24_h",
            "acumulado_chuva_96_h",
            "datetime",
            "year",
            "month",
        ]  # , "horario", "data_particao"]
        self.numeric_cols = [
            "acumulado_chuva_15_min",
            "acumulado_chuva_1_h",
            "acumulado_chuva_4_h",
            "acumulado_chuva_24_h",
            "acumulado_chuva_96_h",
        ]
        self.instrument = {"instrument": "Rain Gauge"}
        self.variables = {"variable": "precipitation (mm/h)"}


class DataPreprocessor:
    """
    ADD
    """

    def __init__(self, station_type):
        # self.staged_path = staged_path
        # self.curated_path = curated_path
        # self._logger = Logger.get_logger()
        # self._data_lake_handler = DataLakeHandler()
        self.station_type = station_type
        self.columns = None
        self.read_cols = None
        self.numeric_cols = None
        self._columns_rename_map = None
        self._variable_limits = None
        self._instrument = None

    def run(self):
        """
        ADD
        """
        # self._logger.info(f"Running {self.__class__.__name__}...")
        print(f"Running {self.__class__.__name__}...")

    def _load_data(self):
        """
        ADD
        """
        # ddf = self._data_lake_handler.read_parquet(self.staged_path, columns=columns)
        # df = ddf.compute()
        dfr = pd.read_csv(f"{self.station_type}_bq.csv", usecols=self.columns)
        print(f"\nFile opened\n{dfr.iloc[0]}")
        print(f"\nFile opened\n{dfr.iloc[1]}")
        print(f"\nFile opened\n{dfr.iloc[2]}")
        return dfr

    def _cast_columns(self, dfr: pd.DataFrame):
        """
        Casting columns to right format, create datetime column and fix HBV.
        """

        dfr["id_estacao"] = dfr["id_estacao"].astype("category")
        for col in self.numeric_cols:
            print(f"Casting {col} column...")
            dfr[col] = pd.to_numeric(dfr[col], errors="coerce")
        print("Building string datetime column...")
        print(f"isna: {dfr.isna().sum()}")
        dfr["datetime"] = dfr["data_particao"].astype(str) + " " + dfr["horario"].astype(str)
        print(dfr.iloc[0])
        print(dfr.iloc[1])
        print(f"isna: {dfr.isna().sum()}")
        print("Casting datetime column...")
        dfr["datetime"] = pd.to_datetime(
            dfr["datetime"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
        )
        print("")
        print(dfr.iloc[0])
        print(dfr.iloc[1])
        dfr.drop(["data_particao", "horario"], axis=1, inplace=True)
        # if not args.skip_hbv_fixer:
        #     print(f"Fixing HBV indicator...")
        #     dfr = self._hbv_fixer.fix(dfr)
        # print('Localizing datetime column to "America/Sao_Paulo" timezone...')
        # dfr["datetime"] = dfr["datetime"].dt.tz_localize("America/Sao_Paulo", ambiguous=dfr["HBV"])
        # dfr.drop("HBV", axis=1, inplace=True)
        return dfr

    def _drop_duplicates(self, dfr: pd.DataFrame):
        print("Start removing duplicates")
        n_rows = dfr.shape[0]
        dfr = dfr.drop_duplicates()
        dfr_temporal_duplicates = dfr[dfr.duplicated(subset=["id_estacao", "datetime"], keep=False)].copy()
        print(f"Total temporal duplicates: {dfr_temporal_duplicates.shape[0]}")
        feature_columns = list(self._variable_limits.keys())
        print(f"feature_columns: {feature_columns}")
        dfr_to_kept = dfr_temporal_duplicates.dropna(subset=feature_columns, how="all")
        print(f"dfr_to_kept: {dfr_to_kept.shape[0]}")
        if dfr_to_kept.duplicated(subset=["datetime", "id_estacao"]).sum() > 0:
            # self._logger.warning(
            #     "There are temporal duplicates with different precipitation values"
            # )
            print("There are temporal duplicates with different precipitation values")
            # print(dfr_to_kept.sort_values(by=["datetime", "id_estacao"]).iloc[0])
            # print(dfr_to_kept.sort_values(by=["datetime", "id_estacao"]).iloc[1])
            # print(dfr_to_kept.sort_values(by=["datetime", "id_estacao"]).iloc[2])
        dfr_to_drop = dfr_temporal_duplicates[dfr_temporal_duplicates[feature_columns].isna()]
        print(f"Total drops: {dfr_to_drop.shape[0]}")
        dfr = dfr[~dfr.index.isin(dfr_to_drop.index)]
        current_n_rows = dfr.shape[0]
        n_removed_rows = n_rows - current_n_rows
        print("End removing duplicates")
        return dfr, n_removed_rows

    def _remove_inconsistent_values(self, dfr: pd.DataFrame):
        print("Start removing inconsistent values")
        variables = self._variable_limits.keys()
        for variable in variables:
            variable_limits = self._variable_limits[variable]
            c_1 = dfr[variable] < variable_limits["min"]
            c_2 = dfr[variable] > variable_limits["max"]
            dfr.loc[c_1 | c_2, variable] = np.nan
        print("End removing inconsistent values")
        return dfr

    def _add_latitude_longitude(self, dfr: pd.DataFrame):
        filepath = "landing/instruments_info/alertario_stations.parquet"
        columns = ["estacao_desc", "latitude", "longitude"]
        # dfr_stations = self._data_lake_handler.read_parquet(filepath, columns=columns).compute()
        # dfr = dfr.merge(dfr_stations, left_on="station", right_on="estacao_desc", how="left")
        dfr = dfr.drop(columns=["estacao_desc"])
        return dfr

    # def _df_to_arrow(self, df: pd.DataFrame):
    # """
    # Save treated data as parquet using partitions
    # """
    #     self._logger.debug(f'Building partition cols and metadata')
    #     df = df.reset_index(drop=True)
    #     df = self._set_partition_cols(df)
    #     attrs = self._build_metadata()
    #     self._logger.debug(f'Converting dataframe to arrow table')
    #     table = pa.Table.from_pandas(df)
    #     table = table.cast(
    #         pa.schema(
    #             table.schema,
    #             metadata=attrs
    #         )
    #     )
    #     return table

    # def _set_partition_cols(self, df:pd.DataFrame):
    #     df['year'] = df['datetime'].dt.year
    #     df['month'] = df['datetime'].dt.month
    #     return df

    def _build_metadata(self):
        attrs = {
            "naming_authority": "Alerta Rio",
            "timezone": "UTC",
            "instrument": self._instrument,
        }
        attrs.update(self._build_variables())
        return attrs

    def _build_variables(self):
        raise NotImplementedError


# # class WeatherStationDataPreprocessor(DataPreprocessor):

# #     def __init__(self, staged_path, curated_path) -> None:
# #         super().__init__(staged_path, curated_path)
# #         self._instrument = 'weather station'
# #         self._columns = [
# #             'station',
# #             'datetime',
# #             'precipitation',
# #             'wind_dir',
# #             'wind_speed',
# #             'temperature',
# #             'pressure',
# #             'humidity',
# #         ]
# #         self._columns_rename_map = None
# #         self._variable_limits = self._build_variable_limits()


# #     def run(self):
# #         self._logger.info(f"Loading data from {self.staged_path}")
# #         df = self._load_data()
# #         self._logger.info('Dropping duplicates')
# #         df, n_removed_rows = self._drop_duplicates(df)
# #         self._logger.info(f'Dropped {n_removed_rows} rows')
# #         self._logger.info('Removing inconsistent values')
# #         df = self._remove_inconsistent_values(df)
# #         self._logger.info('Removing wind direction values for null wind speed')
# #         df = self._remove_inconsistent_wind_dir(df)
# #         self._logger.info('Converting datetime to UTC')
# #         df['datetime'] = df['datetime'].dt.tz_convert('UTC')
# #         self._logger.info('Encoding cyclic wind')
# #         df['wind_u'], df['wind_v'] = cyclic_wind_encoding(df['wind_speed'], df['wind_dir'])
# #         self._logger.info('Encoding cyclic time')
# #         df['hour_sin'], df['hour_cos'] = cyclic_time_encoding(df['datetime'])
# #         df['month_sin'], df['month_cos'] = cyclic_month_encoding(df['datetime'])
# #         self._logger.info('Adding latitude and longitude')
# #         df = self._add_latitude_longitude(df)
# #         self._logger.info('Preparing data to be stored')
# #         table = self._df_to_arrow(df)
# #         self._logger.info(f'Storing data in {self.curated_path}')
# #         self._data_lake_handler.store_data(table, self.curated_path)


# #     def _remove_inconsistent_wind_dir(self, df:pd.DataFrame):
# #         c1 = df['wind_dir'].notna()
# #         c2 = df['wind_speed'].isna()
# #         df.loc[c1 & c2, 'wind_dir'] = np.nan
# #         return df


# #     def _build_variable_limits(self):
# #         return {
# #             'precipitation': {
# #                 'min': 0.0,
# #                 'max': 70.0,
# #             },
# #             'wind_dir': {
# #                 'min': 0.0,
# #                 'max': 360.0,
# #             },
# #             'wind_speed': {
# #                 'min': 0.0,
# #                 'max': 120.0,
# #             },
# #             'temperature': {
# #                 'min': 0.0,
# #                 'max': 50.0,
# #             },
# #             'pressure': {
# #                 'min': 800.0,
# #                 'max': 1200.0,
# #             },
# #             'humidity': {
# #                 'min': 0.0,
# #                 'max': 100.0,
# #             },
# #         }


# #     def _build_variables(self):
# #         values = [
# #             'station - station name',
# #             'datetime - UTC datetime of the measurement',
# #             'precipitation (mm/15min)',
# #             'wind_dir (degrees)',
# #             'wind_speed (km/h)',
# #             'temperature (degrees Celsius)',
# #             'pressure (hPa)',
# #             'humidity (%)',
# #             'wind_u (cyclic U component from the wind)',
# #             'wind_v (cyclic V component from the wind)',
# #             'hour_sin (sine encoding of the time of day)',
# #             'hour_cos (cosine encoding of the time of day)',
# #             'month_sin (sine encoding of the month of year)',
# #             'month_cos (cosine encoding of the month of year)',
# #             'latitude (degrees)',
# #             'longitude (degrees)',
# #         ]
# #         keys = [f'variable_{i}' for i in range(1,len(values)+1)]
# #         return dict(zip(keys, values))


class RainGaugeDataPreprocessor(DataPreprocessor):
    """
    ADD
    """

    def __init__(self, station_type) -> None:
        super().__init__(station_type)
        self._instrument = "rain gauge"
        self.columns = [
            "id_estacao",
            "acumulado_chuva_15_min",
            "acumulado_chuva_1_h",
            "acumulado_chuva_4_h",
            "acumulado_chuva_24_h",
            "acumulado_chuva_96_h",
            "horario",
            "data_particao"]
        self.numeric_cols = [
            "acumulado_chuva_15_min",
            "acumulado_chuva_1_h",
            "acumulado_chuva_4_h",
            "acumulado_chuva_24_h",
            "acumulado_chuva_96_h",
        ]
        self.instrument = {"instrument": "Rain Gauge"}
        self.variables = {"variable": "precipitation (mm/h)"}
        self._columns_rename_map = {
            "acumulado_chuva_15_min": "precipitation",
        }
        self._variable_limits = {
            "precipitation": {
                "min": 0.0,
                "max": 70.0,
            },
        }

    def run(self):
        try:
            # self._logger.info(f"Loading data from {self.staged_path}")
            print(f"Loading data from {self.station_type}")
            dfr = self._load_data()
            print(f"dtypes {dfr.dtypes}")
            dfr = dfr.reset_index(drop=True)
            # code to stage data for HBV ambiguous fixer
            # station_type = self.remote_prefix.split('/')[0]
            # dfr.to_pickle(f'data/hbv_fixer_{station_type}_YYYY.pkl')
            print("Casting columns")
            dfr = self._cast_columns(dfr)
            print("Creating year and month column")
            if self._columns_rename_map:
                print(f"Renaming columns {self._columns_rename_map}")
                dfr = dfr.rename(columns=self._columns_rename_map)
            dfr.reset_index(drop=True, inplace=True)
            dfr["year"] = dfr["datetime"].dt.year
            dfr["month"] = dfr["datetime"].dt.month
            print(dfr.shape)
            attrs = {"naming_authority": "Alerta Rio", "timezone": "America/Sao_Paulo"}
            attrs.update(self.instrument)
            attrs.update(self.variables)
            print("Dropping duplicates")
            # self._logger.info("Dropping duplicates")
            dfr, n_removed_rows = self._drop_duplicates(dfr)
            print(f"Dropped {n_removed_rows} rows")
            print(dfr.shape)
            print("Removing inconsistent values")
            dfr = self._remove_inconsistent_values(dfr)
            print(dfr.shape)
            print("Converting datetime to UTC")
            print(f"{dfr.iloc[0]}")
            dfr["datetime"] = dfr["datetime"].dt.tz_localize("America/Sao_Paulo")
            dfr["datetime"] = dfr["datetime"].dt.tz_convert("UTC")

            print("Encoding cyclic time")
            dfr["hour_sin"], dfr["hour_cos"] = cyclic_time_encoding(dfr["datetime"])
            dfr["month_sin"], dfr["month_cos"] = cyclic_month_encoding(dfr["datetime"])

            # ??? descomentar e adicionar enrtada de dados
            # print("Adding latitude and longitude")
            # dfr = self._add_latitude_longitude(dfr)

            # print("Preparing data to be stored")
            # table = self._dfr_to_arrow(dfr)

            # print(f"Storing data in {self.curated_path}")
            # self._data_lake_handler.store_data(table, self.curated_path)

            save_path = f"{self.station_type}_optimized.csv"
            print(f"File before saving\n{dfr.iloc[0]}")
            print(f"Saving file on path {save_path}.")
            dfr.to_csv(save_path, index=False)
        except Exception as error:
            print(f"Failed to process {self.station_type}: {error}")

    def _build_variables(self):
        values = [
            "station - station name",
            "datetime - UTC datetime of the measurement",
            "precipitation (mm/15min)",
            "hour_sin - sine encoding of the time of day",
            "hour_cos - cosine encoding of the time of day",
            "month_sin - sine encoding of the month of year",
            "month_cos - cosine encoding of the month of year",
            "latitude (degrees)",
            "longitude (degrees)",
        ]
        keys = [f"variable_{i}" for i in range(1, len(values) + 1)]
        return dict(zip(keys, values))


class ETL:
    """
    Responsable for iniciating transform and preprocessin on Alertario data.
    """

    def __init__(self) -> None:
        # self._logger = Logger.get_logger()
        # self._dlc = DataLakeWrapper().get_client()
        self.nada = None

    def run(self):
        """
        ADD
        """
        print("Script initialized")
        if not any([args.transform, args.preprocessing]):
            print("No parameters passed, performing all steps")
            args.transform = args.preprocessing = True
        self._transform() if args.transform else None
        self._preprocessing() if args.preprocessing else None
        print("Script finished")

    def _transform(self):
        """
        Get data from zip and organize it adding station name as a column
        ?? Posso elimianr essa parte de anos?
        """
        # if args.years:
        #     try:
        #         for year in args.years:
        #             if len(str(year)) != 4 or year < 1970:
        #                 message = f"Invalid year. Years must be a list of 4-digit number equal to or greater than 1970. Received: {args.years}"
        #                 raise Exception(message)
        #         years = sorted(args.years)
        #     except Exception as error:
        #         print(error)
        #         exit()
        # else:
        #     years = []
        if args.station_type not in ["rain_gauge", "weather_station"]:
            raise Exception(
                f"Invalid station type. Must be rain_gauge or weather_station. Received: {args.station_type}"
            )
        station_type = args.station_type
        print(f"Performing transform step. Station type: {station_type}")
        ###### remover comentário
        # Optimizer = (
        #     RainGaugeDataOptimizer if station_type == "rain_gauge" else WeatherStationDataOptimizer
        # )
        Optimizer = RainGaugeDataOptimizer if station_type == "rain_gauge" else None
        optimizer = Optimizer(
            local_data_path="data/alertario",
            station_type=station_type,
        )
        # print("Getting files from data lake...")
        # optimizer.get_objects_from_lake(years)

        print("Processing...")
        optimizer.process_objects()

        print("Transform step finished.")

    def _preprocessing(self):
        """
        ADD
        """
        if args.station_type not in ["rain_gauge", "weather_station"]:
            raise Exception(
                f"Invalid station type. Must be rain_gauge or weather_station. Received: {args.station_type}"
            )
        station_type = args.station_type
        print(f"Performing preprocessing step. Station type: {station_type}")
        ###### remover comentário
        # Preprocessor = (
        #     RainGaugeDataPreprocessor
        #     if station_type == "rain_gauge"
        #     else WeatherStationDataPreprocessor
        # )
        Preprocessor = RainGaugeDataPreprocessor if station_type == "rain_gauge" else None
        preprocessor = Preprocessor(
            station_type
            # f"staged/{station_type}/alertario",
            # f"curated/{station_type}/alertario",
        )
        preprocessor.run()


def parameter_parser():
    """
    ADD
    """
    description = "Script to perform ETL on Alerta Rio Stations data. \n \
        Accepts [-t, -p] arguments to perform transform and preprocessing respectively. \n \
        If no arguments are passed, execute all ETL steps. "

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    # parser = Logger.add_log_parameters(parser, os.path.basename(__file__))

    parser.add_argument(
        "-t", "--transform", action="store_true", help="Perform light transform process."
    )
    parser.add_argument(
        "-p", "--preprocessing", action="store_true", help="Perform heavy transform process."
    )
    parser.add_argument(
        "-s",
        "--station_type",
        nargs="?",
        default="rain_gauge",
        help="Type of station to be processed. \
Possible values: [rain_gauge, weather_station]. Default: [rain_gauge]",
    ),
    parser.add_argument(
        "-y",
        "--years",
        nargs="+",
        type=int,
        help="List of 4-digit number equal to or greater than 1970. \
If not passed, process all available years.",
    )
    parser.add_argument("--skip_hbv_fixer", action="store_true", help="Does not fix HBV column")
    parser.add_argument("dataset1", type=str, help="Description of dataset1")
    # parser.add_argument("dataset2", type=str, help="Description of dataset2")
    return parser.parse_args()


def main():
    """
    ADD
    """
    ETL().run()


if __name__ == "__main__":
    # load_dotenv("config/.env")
    args = parameter_parser()
    # Logger.init(filename=args.logfile, level=args.loglevel, verbose=args.verbose)
    main()
