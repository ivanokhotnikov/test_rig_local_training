import argparse
import logging
import os
import re
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO)


def read_raw_data(raw_data_path: str, all_raw_data_file: str) -> None:
    """
    The read_raw_data function reads all the raw data files from a given directory,
    concatenates them into one dataframe and uploads it to the interim storage.


    Args:
        raw_data_path: str: Specify the path to the folder
        all_raw_data_file: str: Specify the path to the file

    Returns:
        None
    """
    start = datetime.now()
    all_raw_df = pd.DataFrame()
    units = []
    for file in os.listdir(raw_data_path):
        logging.info(f'Reading {file} from {raw_data_path}')
        try:
            if file.endswith('.csv') and 'RAW' in file:
                file_raw_df = pd.read_csv(os.path.join(raw_data_path, file),
                                          header=0,
                                          index_col=False)
            elif (file.endswith('.xlsx')
                  or file.endswith('.xls')) and 'RAW' in file:
                file_raw_df = pd.read_excel(os.path.join(raw_data_path, file),
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
        file_raw_df['UNIT'] = unit
        file_raw_df['TEST'] = int(units.count(unit))
        all_raw_df = pd.concat((all_raw_df, file_raw_df), ignore_index=True)
        del file_raw_df
    try:
        all_raw_df.sort_values(by=['UNIT', 'TEST'],
                               inplace=True,
                               ignore_index=True)
        logging.info(f'Final dataframe sorted')
    except:
        logging.info('Cannot sort dataframe')
    os.makedirs('/'.join(all_raw_data_file.split('/')[:-1]), exist_ok=True)
    all_raw_df.to_csv(all_raw_data_file, index=False)
    logging.info('All raw dataframe saved')
    logging.info(
        f'Elapsed time {os.path.basename(__file__)} {datetime.now() - start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--all_raw_data_file', type=str)
    args = parser.parse_args()
    read_raw_data(raw_data_path=args.raw_data_path,
                  all_raw_data_file=args.all_raw_data_file)
