import numpy as np
import pandas as pd


def create_sequences(df: pd.DataFrame, lookback: int, target: str,
                     inference: bool, univariate: bool) -> tuple | np.ndarray:
    """
    The create_sequences function creates a sequence of data for the model to train on. 
    The function takes in a DataFrame, the number of lookback periods, and the target column name. 
    If univariate is set to True, then only one feature will be used (the target). If it is False, then all features will be used. 
    The function returns two numpy arrays: x_train and y_train.
    
    Args:
        df: pd.DataFrame: Pass the dataframe to the function
        lookback: int: Define the number of timesteps in each sample
        target: str: Specify the column name of the target variable
        inference: bool: Distinguish between training and inference
        univariate: bool: Determine whether the model is univariate or multivariate
    
    Returns:
        A tuple if univariate is true, otherwise it returns a numpy array
    """
    x_train, y_train = [], []
    if univariate:
        for i in range(lookback, len(df)):
            x_train.append(df.iloc[i - lookback:i][target])
            y_train.append(df.iloc[i][target])
        x_train = np.expand_dims(x_train, axis=-1)
    else:
        for i in range(lookback, len(df)):
            x_train.append(df.iloc[i - lookback:i])
            y_train.append(df.iloc[i][target])
        x_train = np.stack(x_train)
    y_train = np.expand_dims(y_train, axis=-1)
    if inference:
        return x_train
    return x_train, y_train