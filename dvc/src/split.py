import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd
from dvc.api import params_show

logging.basicConfig(level=logging.INFO)


def split(interim_data_file: str, train_data_file: str, test_data_file: str,
          forecast_features_file: list) -> None:
    """
    The split function splits the data into train and test sets.
    The function takes in an interim data path, a train data path, and a test data path as arguments.
    It then reads the interim dataset from the interim data path argument,
    splits it into training and testing datasets based on the size of the training set (as determined by params_show).
    Then it writes those datasets to their respective paths.

    Args:
        interim_data_file: str: Specify the path to the interim data
        train_data_file: str: Specify the path where the train data will be stored
        test_data_file: str: Specify the path where the test data will be stored

    Returns:
        None
    """
    start = datetime.now()
    params = params_show(stages='split')
    train_size = params['split']['train_size']
    with open(forecast_features_file, 'r') as fp:
        forecast_features = json.load(fp)
    interim_df = pd.read_parquet(interim_data_file, columns=forecast_features)
    train = interim_df.iloc[:int(len(interim_df) * train_size)]
    test = interim_df.iloc[int(len(interim_df) * train_size):]
    os.makedirs('/'.join(train_data_file.split('/')[:-1]), exist_ok=True)
    train.to_parquet(train_data_file, index=False)
    logging.info('Train dataset saved')
    os.makedirs('/'.join(test_data_file.split('/')[:-1]), exist_ok=True)
    test.to_parquet(test_data_file, index=False)
    logging.info('Test dataset saved')
    logging.info(
        f'Elapsed time {os.path.basename(__file__)} {datetime.now() - start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim_data_file', type=str)
    parser.add_argument('--forecast_features_file', type=str)
    parser.add_argument('--train_data_file', type=str)
    parser.add_argument('--test_data_file', type=str)
    args = parser.parse_args()
    split(interim_data_file=args.interim_data_file,
          forecast_features_file=args.forecast_features_file,
          train_data_file=args.train_data_file,
          test_data_file=args.test_data_file)
