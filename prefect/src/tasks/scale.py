import os

import joblib
import pandas as pd
from prefect import task
from prefect.tasks import task_input_hash
from sklearn.preprocessing import StandardScaler


@task(cache_key_fn=task_input_hash, refresh_cache=True)
def scale(df: pd.DataFrame, processed_data_path: str, artifacts_path: str,
          test: bool) -> tuple | pd.DataFrame:
    """
    The scale function takes a DataFrame and returns a scaled version of it.
    If test is True, then the scaler will be loaded from artifacts_path instead
    of being created.

    Args:
        df: pd.DataFrame: Pass the dataframe to be scaled
        processed_data_path: str: Specify the path to save the processed data
        artifacts_path: str: Store the scaler
        test: bool: Indicate whether the function is called during training or testing

    Returns:
        The scaler and the scaled dataframe if train, scaled dataframe only if test
    """
    if not test:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values)
        os.makedirs(artifacts_path, exist_ok=True)
        joblib.dump(scaler, os.path.join(artifacts_path, 'scaler.joblib'))
    else:
        scaler = joblib.load(os.path.join(artifacts_path, 'scaler.joblib'))
        scaled = scaler.transform(df.values)
    os.makedirs(processed_data_path, exist_ok=True)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    scaled_df.to_parquet(os.path.join(
        processed_data_path,
        f'scaled_{"train" if not test else "test"}.parquet'),
                         index=False)
    return scaler, scaled_df
