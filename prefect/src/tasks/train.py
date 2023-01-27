import os

import numpy as np
import pandas as pd
from prefect.tasks import task_input_hash
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
from utils.create_sequences import create_sequences

from prefect import task


@task(cache_key_fn=task_input_hash, refresh_cache=False)
def train(df: pd.DataFrame,
          target: str,
          folds: int,
          lookback: int,
          univariate: bool,
          lstm_units: int,
          learning_rate: float,
          batch_size: int,
          val_split: float,
          epochs: int,
          patience: int,
          verbose: int,
          timestamp: str | None = None) -> tuple:
    """
    The train function trains a model on the data provided.

    Args:
        df: pd.DataFrame: Pass the dataframe with all the features
        target: str: Define the column name of the target variable
        folds: int: Define the number of folds to be used in cross-validation
        lookback: int: Define the number of time steps to look back in order to predict the next value
        univariate: bool: Create the sequences for training and validation
        lstm_units: int: Define the number of units in each lstm layer
        learning_rate: float: Control the magnitude of the updates to weights during training
        batch_size: int: Specify the number of samples to work through before updating the internal model parameters
        val_split: float: Specify the fraction of the training data to be used as validation data
        epochs: int: Specify the number of epochs to train for
        patience: int: Stop the training early if the loss does not improve after a given number of epochs
        verbose: int: Control the amount of logging information printed during training

    Returns:
        A tuple with two elements: model and val_metrics
    """
    tscv = TimeSeriesSplit(n_splits=folds)
    val_loss, val_rmse = [], []
    callbacks = [
        keras.callbacks.EarlyStopping(patience=patience,
                                      monitor='val_loss',
                                      mode='min',
                                      verbose=verbose,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.75,
                                          patience=patience // 2,
                                          verbose=verbose,
                                          mode='min')
    ]
    for fold, (train_index, val_index) in enumerate(tscv.split(df), start=1):
        if timestamp is not None:
            callbacks.append(
                keras.callbacks.TensorBoard(log_dir=os.path.join(
                    'tensorboards', target, timestamp, str(fold)),
                                            histogram_freq=1,
                                            write_graph=True,
                                            write_images=True,
                                            update_freq='epoch'))
        x_train, y_train = create_sequences(df=df.iloc[train_index],
                                            lookback=lookback,
                                            univariate=univariate,
                                            target=target,
                                            inference=False)
        model = keras.models.Sequential(name='forecaster')
        model.add(
            keras.layers.LSTM(lstm_units,
                              input_shape=(x_train.shape[1], x_train.shape[2]),
                              return_sequences=False))
        model.add(keras.layers.Dense(1))
        model.compile(
            loss=keras.losses.mean_squared_error,
            metrics=keras.metrics.RootMeanSquaredError(),
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
        history = model.fit(x_train,
                            y_train,
                            shuffle=False,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=val_split,
                            verbose=verbose,
                            callbacks=callbacks)
        x_val, y_val = create_sequences(df=df.iloc[val_index],
                                        lookback=lookback,
                                        univariate=univariate,
                                        target=target,
                                        inference=False)
        val_loss_in_fold, val_rmse_in_fold = model.evaluate(
            x=x_val, y=y_val, batch_size=batch_size, verbose=verbose)
        val_loss.append(val_loss_in_fold)
        val_rmse.append(val_rmse_in_fold)
    val_metrics = {
        'val_loss_mean': np.mean(val_loss),
        'val_loss_std': np.std(val_loss),
        'val_rmse_mean': np.mean(val_rmse),
        'val_rmse_std': np.std(val_rmse)
    }
    return model, val_metrics
