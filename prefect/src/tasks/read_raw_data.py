import os
import re

import pandas as pd

from prefect import task


def cache_key_from_files_num(context, parameters):
    return len(os.listdir(parameters['raw_data_path']))


@task(cache_key_fn=cache_key_from_files_num, refresh_cache=False)
def read_raw_data(raw_data_path: str, interim_data_path: str) -> pd.DataFrame:
    """
    The read_raw_data function reads all the raw data files from a specified directory,
    concatenates them into a single dataframe and saves it to an interim data directory.
    It also returns the concatenated dataframe.

    Args:
        raw_data_path: str: Specify the path to the raw data files
        interim_data_path: str: Specify the path to save the all_raw_data

    Returns:
        A pandas dataframe with all the raw data files read
    """
    all_raw_df = pd.DataFrame()
    units = []
    for file in os.listdir(raw_data_path):
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
                continue
            unit = int(re.split(r'_|-', file.lstrip('-/HYDhyd0'))[0][-4:])
        except:
            continue
        units.append(unit)
        file_raw_df['UNIT'] = unit
        file_raw_df['TEST'] = int(units.count(unit))
        all_raw_df = pd.concat((all_raw_df, file_raw_df), ignore_index=True)
        del file_raw_df
    all_raw_df.sort_values(by=['UNIT', 'TEST'],
                           inplace=True,
                           ignore_index=True)
    os.makedirs(interim_data_path, exist_ok=True)
    all_raw_df.to_csv(os.path.join(interim_data_path, 'all_raw.csv'),
                      index=False)
    return all_raw_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str)
    parser.add_argument('--interim', type=str)
    args = parser.parse_args()
    df = read_raw_data.fn(raw_data_path=args.raw,
                          interim_data_path=args.interim)
