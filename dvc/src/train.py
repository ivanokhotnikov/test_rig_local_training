import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from dvc.api import params_show
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras

from create_sequences import create_sequences

logging.basicConfig(level=logging.INFO)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])
    logical_gpus = tf.config.list_logical_devices('GPU')


def train(scaled_data_file: str, target: str, val_metrics_file: str,
          model_path: str) -> None:
    """
    The train function trains a model on the data.
    
    Args:
        scaled_data_file: str: Specify the path to the parquet file containing the scaled data
        target: str: Specify the target column name
        val_metrics_file: str: Store the metrics of each fold
        model_path: str: Save the model in a file
    
    Returns:
        None
    """
    start = datetime.now()
    params = params_show(stages='train')
    tscv = TimeSeriesSplit(n_splits=params['train']['folds'])
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    val_loss, val_rmse = [], []
    train_df = pd.read_parquet(scaled_data_file)
    for fold, (train_index, val_index) in enumerate(tscv.split(train_df),
                                                    start=1):
        logging.info(f'Fold {fold}:')
        x_train, y_train = create_sequences(
            df=train_df.iloc[train_index],
            lookback=params['train']['lookback'],
            univariate=params['train']['univariate'],
            target=target,
            inference=False)
        model = keras.models.Sequential(name='forecaster')
        model.add(
            keras.layers.LSTM(params['train']['lstm_units'],
                              input_shape=(x_train.shape[1], x_train.shape[2]),
                              return_sequences=False))
        model.add(keras.layers.Dense(1))
        model.compile(loss=keras.losses.mean_squared_error,
                      metrics=keras.metrics.RootMeanSquaredError(),
                      optimizer=keras.optimizers.RMSprop(
                          learning_rate=params['train']['learning_rate']))
        history = model.fit(
            x_train,
            y_train,
            shuffle=False,
            epochs=params['train']['epochs'],
            batch_size=params['train']['batch_size'],
            validation_split=params['train']['val_size'],
            verbose=params['train']['verbose'],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=params['train']['patience'],
                    monitor='val_loss',
                    mode='min',
                    verbose=params['train']['verbose'],
                    restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.75,
                    patience=params['train']['patience'] // 2,
                    verbose=params['train']['verbose'],
                    mode='min'),
                keras.callbacks.TensorBoard(log_dir=os.path.join(
                    'logs', 'tensorboards', target, timestamp, str(fold)),
                                            histogram_freq=1,
                                            write_graph=True,
                                            write_images=True,
                                            update_freq='epoch')
            ])
        x_val, y_val = create_sequences(
            df=train_df.iloc[val_index],
            lookback=params['train']['lookback'],
            univariate=params['train']['univariate'],
            target=target,
            inference=False)
        val_loss_in_fold, val_rmse_in_fold = model.evaluate(
            x=x_val,
            y=y_val,
            batch_size=params['train']['batch_size'],
            verbose=params['train']['verbose'])
        val_loss.append(val_loss_in_fold)
        val_rmse.append(val_rmse_in_fold)
    val_metrics = {
        'val_loss_mean': np.mean(val_loss),
        'val_loss_std': np.std(val_loss),
        'val_rmse_mean': np.mean(val_rmse),
        'val_rmse_std': np.std(val_rmse)
    }
    with open(val_metrics_file, 'w') as fp:
        json.dump(val_metrics, fp)
    model.save(model_path)
    logging.info(
        f'Elapsed time {os.path.basename(__file__)}@{target} {datetime.now() - start}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaled_data_file', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--val_metrics_file', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    train(scaled_data_file=args.scaled_data_file,
          target=args.target,
          val_metrics_file=args.val_metrics_file,
          model_path=args.model)
