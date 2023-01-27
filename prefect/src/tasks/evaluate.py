from datetime import datetime

import keras
import pandas as pd
from prefect import task
from prefect.tasks import task_input_hash

from utils import create_sequences


@task(cache_key_fn=task_input_hash, refresh_cache=False)
def evaluate(df: pd.DataFrame, model: keras.models.Model, lookback: int,
             batch_size: int, univariate: bool, target: str,
             verbose: int) -> dict:
    """
    The evaluate function evaluates the model on the test data.
    It returns a dictionary with two keys: `test_loss` and `test_rmse`.
    The values of each key are scalar values that give the loss and RMSE on
    the test set, respectively.

    Args:
        df: pd.DataFrame: Pass the dataframe that is used for training and testing
        model: keras.models.Model: Evaluate the model
        lookback: int: Define how many timesteps back the input data should go
        batch_size: int: Define the number of samples per gradient update
        univariate: bool: Determine whether the model is univariate or multivariate
        target: str: Specify the column name of the target variable
        verbose: int: Specify the verbosity of the model

    Returns:
        A dictionary with two values: test_loss, test_rmse
    """
    x_test, y_test = create_sequences(df=df,
                                      lookback=lookback,
                                      univariate=univariate,
                                      target=target,
                                      inference=False)
    test_loss, test_rmse = model.evaluate(x=x_test,
                                          y=y_test,
                                          batch_size=batch_size,
                                          verbose=verbose)
    return {'test_loss': test_loss, 'test_rmse': test_rmse}
