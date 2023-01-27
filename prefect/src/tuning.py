import json
import os

import optuna
import pandas as pd
from dotenv import load_dotenv
from optuna.integration.mlflow import MLflowCallback
from tasks import evaluate, train
from training import get_arguments

from prefect import flow


@flow(name='tuning workflow')
def tuning_workflow(trial):
    model, val_metrics = train(
        df=scaled_train_df,
        target=feature,
        folds=args['folds'],
        lookback=args['lookback'],
        univariate=args['univariate'],
        lstm_units=trial.suggest_int('lstm_units', 5, 50),
        learning_rate=trial.suggest_float('learning_rate',
                                          1e-6,
                                          1e-4,
                                          log=True),
        batch_size=args['batch_size'],
        val_split=args['val_split'],
        epochs=args['epochs'],
        patience=args['patience'],
        verbose=args['verbose'])
    test_metrics = evaluate(df=scaled_test_df,
                            model=model,
                            lookback=args['lookback'],
                            batch_size=args['batch_size'],
                            univariate=args['univariate'],
                            target=feature,
                            verbose=args['verbose'])
    return test_metrics['test_rmse']


if __name__ == '__main__':
    load_dotenv()
    scaled_train_df = pd.read_parquet(
        os.path.join(os.environ['PROCESSED_DATA_PATH'],
                     'scaled_train.parquet'))
    scaled_test_df = pd.read_parquet(
        os.path.join(os.environ['PROCESSED_DATA_PATH'], 'scaled_test.parquet'))
    args = vars(get_arguments())
    with open(
            os.path.join(os.environ['FEATURES_PATH'],
                         'forecast_features.json'), 'r') as fp:
        features = json.load(fp)
    mlflow_callback = MLflowCallback(metric_name=['val_rmse_mean'])
    for feature in features:
        study = optuna.create_study(direction='minimize')
        study.optimize(tuning_workflow,
                       n_trials=100,
                       timeout=1000,
                       callbacks=[mlflow_callback])
