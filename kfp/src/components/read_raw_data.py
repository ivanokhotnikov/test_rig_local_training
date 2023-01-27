from kfp.v2.dsl import Artifact, Dataset, Output, component

from utils.dependencies import OPENPYXL, PANDAS, PYARROW, PYTHON310


@component(base_image=PYTHON310, packages_to_install=[PANDAS, OPENPYXL, PYARROW])
def read_raw_data(raw_data_path: str, features_path: str,
                  interim_data_path: str, interim_data: Output[Dataset],
                  raw_features: Output[Artifact]) -> None:
    """
    The read_raw_data function reads the raw data from the raw_data_path, 
    and uploads it to the pipeline metadata store and an interim data storage. 
    The function also creates a JSON file containing all of the features in 
    the dataset and uploads it to both locations.
    
    Args:
        raw_data_path: str: Specify the path to the raw data files
        features_path: str: Specify the path where the raw features are stored
        interim_data_path: str: Specify the path to store the interim data
        interim_data: Output[Dataset]: Upload the interim data to the pipeline metadata store
        raw_features: Output[Artifact]: Upload the list of features to the pipeline metadata store
    """
    import gc
    import json
    import logging
    import os
    import re

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    final_df = pd.DataFrame()
    units = []
    for file in os.listdir(raw_data_path):
        logging.info(f'Reading {file} from {raw_data_path}')
        try:
            if file.endswith('.csv') and 'RAW' in file:
                current_df = pd.read_csv(os.path.join(raw_data_path, file),
                                         header=0,
                                         index_col=False)
            elif (file.endswith('.xlsx')
                  or file.endswith('.xls')) and 'RAW' in file:
                current_df = pd.read_excel(os.path.join(raw_data_path, file),
                                           header=0,
                                           index_col=False)
            else:
                logging.info(f'{file} is not a valid raw data file')
                continue
        except:
            logging.info(f'Cannot read {file}')
            continue
        logging.info(f'{file} has been read')
        try:
            unit = int(re.split(r'_|-', file.lstrip('-/HYDhyd0'))[0][-4:])
        except ValueError as err:
            logging.info(f'{err}\n. Cannot parse unit from {file}')
            continue
        units.append(unit)
        current_df['UNIT'] = unit
        current_df['TEST'] = int(units.count(unit))
        final_df = pd.concat((final_df, current_df), ignore_index=True)
        del current_df
        gc.collect()
    try:
        final_df.sort_values(by=['UNIT', 'TEST'],
                             inplace=True,
                             ignore_index=True)
        logging.info(f'Final dataframe sorted')
    except:
        logging.info('Cannot sort dataframe')
    final_df.to_csv(interim_data.path + '.csv', index=False)
    logging.info('Interim dataframe uploaded to the pipeline metadata store')
    final_df.to_csv(os.path.join(interim_data_path, 'interim_data.csv'),
                    index=False)
    logging.info('Interim dataframe uploaded to the interim data storage')
    with open(raw_features.path + '.json', 'w') as features_file:
        json.dump(final_df.columns.to_list(), features_file)
    logging.info('Raw features uploaded to the pipeline metadata store')
    with open(os.path.join(features_path, 'raw_features.json'),
              'w') as features_file:
        json.dump(final_df.columns.to_list(), features_file)
    logging.info('Raw features uploaded to the features store')
