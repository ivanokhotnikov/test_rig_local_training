import os

import pandas as pd
from prefect.tasks import task_input_hash

from prefect import task


@task(cache_key_fn=task_input_hash, refresh_cache=False)
def split(df: pd.DataFrame, train_split: float,
          processed_data_path: str) -> tuple:
    """
    The split function splits the data into train and test sets.
    The split function takes a DataFrame, df, as input and returns two DataFrames:
    train_df (the training set) and test_df (the testing set). The split is done by
    randomly selecting rows from the original dataset until enough rows for the
    training set is acquired. The rest of the rows are used for the testing set.

    Args:
        df: pd.DataFrame: Indicate the dataframe to be used
        train_split: float: Specify the percentage of data to be used for training
        processed_data_path: str: Specify the path where to save the processed data

    Returns:
        A tuple with the train and test datasets
    """
    interim_df = df.copy(deep=True)
    train = interim_df.iloc[:int(len(interim_df) * train_split)]
    test = interim_df.iloc[int(len(interim_df) * train_split):]
    os.makedirs(processed_data_path, exist_ok=True)
    train.to_parquet(os.path.join(processed_data_path, 'train.parquet'),
                     index=False)
    test.to_parquet(os.path.join(processed_data_path, 'test.parquet'),
                    index=False)
    return train, test
