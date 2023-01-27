import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from mlflow import log_metric, log_param, start_run
from tasks import evaluate, preprocess, read_raw_data, scale, split, train

from prefect import flow


@flow(name='training workflow')
def training_workflow(train_split: float, folds: int, lookback: int,
                      val_split: float, epochs: int, batch_size: int,
                      patience: int, lstm_units: int, learning_rate: float,
                      verbose: int, univariate: bool):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    all_raw_df = read_raw_data(
        raw_data_path=os.environ['RAW_DATA_PATH'],
        interim_data_path=os.environ['INTERIM_DATA_PATH'])
    forecast_features, interim_df = preprocess(
        df=all_raw_df,
        features_path=os.environ['FEATURES_PATH'],
        interim_data_path=os.environ['INTERIM_DATA_PATH'])
    train_df, test_df = split(
        df=interim_df,
        train_split=train_split,
        processed_data_path=os.environ['PROCESSED_DATA_PATH'])
    scaler, scaled_train_df = scale(
        train_df,
        processed_data_path=os.environ['PROCESSED_DATA_PATH'],
        artifacts_path=os.environ['ARTIFACTS_PATH'],
        test=False)
    _, scaled_test_df = scale(
        test_df,
        processed_data_path=os.environ['PROCESSED_DATA_PATH'],
        artifacts_path=os.environ['ARTIFACTS_PATH'],
        test=True)
    for feature in forecast_features:
        with start_run(run_name=f'{feature}', nested=True):
            model, val_metrics = train(df=scaled_train_df,
                                       target=feature,
                                       folds=folds,
                                       lookback=lookback,
                                       univariate=univariate,
                                       lstm_units=lstm_units,
                                       learning_rate=learning_rate,
                                       batch_size=batch_size,
                                       val_split=val_split,
                                       epochs=epochs,
                                       patience=patience,
                                       timestamp=timestamp,
                                       verbose=verbose)
            with open(
                    os.path.join(os.environ['ARTIFACTS_PATH'],
                                 f'val_metrics_{feature}.json'), 'w') as fp:
                json.dump(val_metrics, fp)
            model.save(
                os.path.join(os.environ['ARTIFACTS_PATH'],
                             f'model_{feature}.h5'))
            for k, v in val_metrics.items():
                log_metric(k, v)
            test_metrics = evaluate(df=scaled_test_df,
                                    model=model,
                                    lookback=lookback,
                                    batch_size=batch_size,
                                    univariate=univariate,
                                    target=feature,
                                    verbose=verbose)
            with open(
                    os.path.join(os.environ['ARTIFACTS_PATH'],
                                 f'test_metrics_{feature}.json'), 'w') as fp:
                json.dump(test_metrics, fp)
            for k, v in test_metrics.items():
                log_metric(k, v)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--lstm_units', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--univariate', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = vars(get_arguments())
    for k, v in args.items():
        log_param(k, v)
    training_workflow(**args)
