import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd
from dvc.api import params_show
from keras.models import load_model

from create_sequences import create_sequences

logging.basicConfig(level=logging.INFO)


def evaluate(scaled_data_file: str, model_path: str, test_metrics_path: str,
             target: str) -> None:
    """
    The evaluate function loads a model and evaluates it against the test data.
    The function saves the results to a file in JSON format.
    
    Args:
        scaled_data_file: str: Pass the path to the scaled data file
        model_path: str: Load the model from disk
        test_metrics_path: str: Store the test metrics
        target: str: Select the target column from the test dataframe
    
    Returns:
        None
    """
    start = datetime.now()
    params = params_show(stages='train')
    test_df = pd.read_parquet(scaled_data_file)
    x_test, y_test = create_sequences(df=test_df,
                                      lookback=params['train']['lookback'],
                                      univariate=params['train']['univariate'],
                                      target=target,
                                      inference=False)
    model = load_model(model_path)
    test_loss, test_rmse = model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=params['train']['batch_size'],
        verbose=params['train']['verbose'])
    test_metrics = {'test_loss': test_loss, 'test_rmse': test_rmse}
    with open(test_metrics_path, 'w') as test_metrics_file:
        json.dump(test_metrics, test_metrics_file)
    logging.info(
        f'Elapsed time {os.path.basename(__file__)}@{target} {datetime.now() - start}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaled_data_file', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--test_metrics', type=str)
    args = parser.parse_args()
    evaluate(scaled_data_file=args.scaled_data_file,
             model_path=args.model,
             target=args.target,
             test_metrics_path=args.test_metrics)
