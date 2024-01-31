# -*- coding: utf-8 -*-
from pipelines.utils.utils import (
    log,
    # to_partitions,
)
from prefect import task
# from prefeitura_rio.pipelines_utils.logging import log # ver


from prefect import Task

from datetime import datetime
import pandas as pd
import pyarrow as pa

class ReadParquetTask(Task):
    def __init__(self, data_lake_handler, **kwargs):
        super().__init__(**kwargs)
        self.data_lake_handler = data_lake_handler

    def run(self, remote_path, columns=None, filters=None):
        ddf = self.data_lake_handler.read_parquet(
            remote_path, 
            columns=columns, 
            filters=filters
        )
        return ddf.compute()


class StoreDataInLakeTask(Task):
    def __init__(self, data_lake_handler, **kwargs):
        super().__init__(**kwargs)
        self.data_lake_handler = data_lake_handler

    def run(self, table, path):
        self.data_lake_handler.store_data_in_lake(table, path)


class DataframeToArrowTask(Task):
    def run(self, df):
        table = pa.Table.from_pandas(df, preserve_index=False)
        return table


class DataIntegratorTask(Task):
    def __init__(self, logger, dl, non_shared_feature_handler, sources, period, **kwargs):
        super().__init__(**kwargs)
        # self.logger = logger
        self.dl = dl
        self.non_shared_feature_handler = non_shared_feature_handler
        self.sources = sources
        self.period = period

    def run(self):
        # Your DataIntegrator logic here
        pass


class DataIntegrator:
    
    def __init__(self) -> None:
        # self._logger = Logger.get_logger()
        self._dl = DataLakeHandler()
        self._feature_handler = self._get_feature_handler(args.non_shared_feature_handler)
        self._datasets = []
        self._all_features = list()
        self._all_non_features = list()
        self._integrated_data = None
        self._all_metadata = self._build_all_metadata()
        self._sources = INSTRUMENTS if args.sources == ['all'] else args.sources

    
    def run(self):
        # self._logger.info('Script initialized')
        self._prepare_data()
        self._integrate_data()
        self._save_data()
        # self._logger.info('Script finished')
    
    
    def _prepare_data(self):
        for string_source in self._sources:
            source = eval(string_source)(self._dl)
            try:
                # self._logger.info(f'Preparing data from source {source.name}...')
                log(f'Preparing data from source {source.name}...')
                source.prepare_data()
                self._datasets.append(source.get_data())
                self._all_features.append(source.get_features())
                self._all_non_features.append(source.get_non_features())
            except:
                message = f'Error while preparing data from source {source.name}. \
Source will be ignored and data integration will continue.'
                # self._logger.error(message)
                log(message)
                continue

    
    def _integrate_data(self):
        # self._logger.info('Integrating data sources...')
        log('Integrating data sources...')
        if len(self._datasets) > 1:
            non_shared_features = self._get_non_shared_elements(self._all_features)
            # self._logger.debug(f'Non shared features: {non_shared_features}')
            log(f'Non shared features: {non_shared_features}')
            non_shared_non_features = self._get_non_shared_elements(self._all_non_features)
            # self._logger.debug(f'Non shared non features: {non_shared_non_features}')
            log(f'Non shared non features: {non_shared_non_features}')
            columns_order = None
            self._integrated_data = pd.DataFrame(data=None)
            for _ in range(len(self._datasets)):
                df = self._datasets.pop(0)
                df = self._feature_handler.handle_features(df, non_shared_features)
                df = self._handle_non_features(df, non_shared_non_features)
                if columns_order is None:
                    columns_order = df.columns
                else:
                    df = df[columns_order]
                self._integrated_data = pd.concat([self._integrated_data, df], ignore_index=True)
                del df
            self._integrated_data = self._integrated_data.reset_index(drop=True)
        else:
            self._integrated_data = self._datasets[0]
            self._datasets = None
    
    
    def _get_non_shared_elements(self, all_elements: list):
        """Returns a set with all elements that are not common to all lists

        Args:
            all_elements (list): A list of lists

        Returns:
            set: a set with all elements that are not common to all lists
        """
        all_elements = list(map(set, all_elements))
        shared_elements = all_elements[0].intersection(*all_elements)
        non_shared_elements = set()
        for s in all_elements:
            non_shared_elements.update(s.difference(shared_elements))
        return non_shared_elements
    
    
    def _get_feature_handler(self, strategy):
        try:
            if not args.non_shared_feature_handler:
                return NoneHandlerStrategy
            handler_strategy = eval(f'{strategy}Strategy')
            return handler_strategy
        except NameError:
            message = f'Class {strategy}Strategy not implemented.'
            # self._logger.error(message)
            log(message)
            exit()
        except Exception as e:
            message = f'Error while instantiating handler strategy'
            # self.logger.error(message)
            # self._logger.error(e)
            log(message)
            log(e)
            exit()
    
    
    def _handle_non_features(self, df: pd.DataFrame, non_shared_non_features: set):
        columns = set(df.columns)
        non_features_to_handle = sorted(non_shared_non_features.difference(columns))
        for non_feature in non_features_to_handle:
            df[non_feature] = None
        return df
    
    
    def _build_all_metadata(self):
        non_features = {
            'institution': 'institution (name of the authority owning the data)',
            'instrument': 'instrument (type of meteorological instrument)',
            'station': 'station (station name)',
            'station_id': 'station_id (station UID)',
            'datetime': 'datetime (UTC datetime of the measurement)', 
            'latitude': 'latitude (degrees)', 
            'longitude': 'longitude (degrees)',
            'altitude': 'altitude (kilometers)',
        }
        features = {
            'horizontal_reflectivity_mean': 'horizontal_reflectivity_mean (mean reflectivity in dBZ)', 
            'dew_point': 'dew_point (dew point temperature in Celsius)', 
            'humidity': 'humidity (instant relative humidity in %)',
            'pressure': 'pressure (instant atmospheric pressure in mB)', 
            'temperature': 'temperature (instant temperature in Celsius)',
            'wind_dir': 'wind_dir (clockwise wind direction in degrees)', 
            'wind_speed': 'wind_speed (hourly wind speed in m/s)', 
            'wind_u': 'wind_u (cyclic U component from the wind)', 
            'wind_v': 'wind_v (cyclic V component from the wind)',
            'precipitation': 'precipitation (hourly precipitation in mm)',
            'hour_sin': 'hour_sin (sine encoding of the time of day)',
            'hour_cos': 'hour_cos (cosine encoding of the time of day)',
            'month_sin': 'month_sin (sine encoding of the month of year)', 
            'month_cos': 'month_cos (cosine encoding of the month of year)',
        }
        return {'features': features, 'non_features': non_features}
    
    
    def _save_data(self):
        # self._logger.info('Preparing data to be stored...')
        log('Preparing data to be stored...')
        table = self._dataframe_to_arrow(self._integrated_data)
        created_at = table.schema.metadata[b'created_at'].decode('utf-8').replace(' ', '_')
        filepath = f'curated/datasets/data_{created_at}.parquet'
        self._dl.store_data_in_lake(table, filepath)
        # self._logger.info(f'Data stored in {filepath}')
        log(f'Data stored in {filepath}')

    
    def _dataframe_to_arrow(self, df:pd.DataFrame):
        # self._logger.debug(f'Converting dataframe to arrow table...')
        log(f'Converting dataframe to arrow table...')
        attrs = self._build_metadata()
        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(
            pa.schema(
                table.schema,
                metadata=attrs
            )
        )
        return table


    def _build_metadata(self):
        attrs = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'param_1': f'sources: {self._sources}', # TODO: filter only sources that were successfully integrated
            'param_2': f'period: {args.period}',
            'param_3': f'non_shared_feature_handler: {args.non_shared_feature_handler}',
        }
        attrs.update(self._build_variables('non_features'))
        attrs.update(self._build_variables('features'))
        return attrs
    
    
    def _build_variables(self, string_category):
        columns = [col for col in self._integrated_data.columns if col in self._all_metadata[string_category].keys()]
        values = [self._all_metadata[string_category][col] for col in columns]
        keys = [f'{string_category}_{i}' for i in range(1,len(values)+1)]
        return dict(zip(keys, values))