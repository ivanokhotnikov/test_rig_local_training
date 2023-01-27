import argparse
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


def scale(data_file: str, scaled_data_file: str, scaler_path: str,
          test: bool) -> None:
    """
    The scale function takes a data path, a scaled data path, and the name of 
    the scaler file. If test is False (default), it will fit the StandardScaler on 
    the training set and save it to disk. If test is True, then it will load the 
    scaler from disk and use that scaler to transform the testing set.

    Args:
        train_data_file: str: Specify the path to the parquet file containing the data
        scaled_train_data_file: str: Specify the path where we want to save the scaled data
        scaler_path: str: Save the scaler object to disk
        test: bool: Determine whether the scaler is fit to the data or not

    Returns:
        None
    """
    start = datetime.now()
    df = pd.read_parquet(data_file)
    if not test:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values)
        os.makedirs('/'.join(scaler_path.split('/')[:-1]), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(df.values)
    os.makedirs('/'.join(scaled_data_file.split('/')[:-1]), exist_ok=True)
    pd.DataFrame(scaled, columns=df.columns).to_parquet(scaled_data_file,
                                                        index=False)
    logging.info(
        f'Elapsed time {os.path.basename(__file__)} {"test" if test else "train"} {datetime.now() - start}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--scaled_data_file', type=str)
    parser.add_argument('--scaler', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    scale(data_file=args.data_file,
          scaled_data_file=args.scaled_data_file,
          scaler_path=args.scaler,
          test=args.test)
