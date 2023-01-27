import argparse
import logging
import os
import sys
from datetime import datetime

from components import (build_features, compare_models, evaluate,
                        import_champion_metrics, import_forecast_features,
                        read_raw_data, split_data, train,
                        upload_model_to_registry)
from kfp.v2 import compiler
from kfp.v2.dsl import Artifact, Condition, ParallelFor, importer, pipeline


@pipeline(name='training-pipeline')
def training_pipeline(project_id: str, region: str, raw_data_path: str,
                      interim_data_path: str, processed_data_path: str,
                      features_path: str, models_path: str,
                      pipelines_path: str, deploy_image: str, timestamp: str,
                      train_data_size: float, lookback: int, lstm_units: int,
                      learning_rate: float, epochs: int, batch_size: int,
                      patience: int) -> None:
    """
    The training_pipeline function is a pipeline that trains and evaluates the model.
    It takes in raw data, builds features, splits the data into train and test sets, 
    trains the model on the training set, evaluates it on test set and uploads it to GCP registry.
    
    Args:
        project_id: str: Specify the project id of the gcp project where you want to run your pipeline
        region: str: Specify the gcp region where the training job will be executed
        raw_data_path: str: Specify the path to the raw data
        interim_data_path: str: Specify the location of the interim data
        processed_data_path: str: Specify the location of processed data
        features_path: str: Specify the path to where the features
        models_path: str: Store the model in a gcs bucket
        deploy_image: str: Set the name of the image that will be used to deploy your model
        timestamp: str: Create a unique name for the experiment
        train_data_size: float: Determine the size of the training data used for each model
        lookback: int: Determine how many time steps to look back when generating features
        lstm_units: int: Set the number of lstm units
        learning_rate: float: Control the magnitude of each gradient descent step
        epochs: int: Determine how many epochs the model will be trained for
        batch_size: int: Define the batch size of the training data
        patience: int: Stop training when the loss does not improve after a specified number of epochs
    """
    interim_features_import = importer(
        artifact_uri='gs://test_rig_features/interim_features.json',
        artifact_class=Artifact).set_display_name('Import interim features')
    read_raw_data_task = read_raw_data(
        raw_data_path=raw_data_path,
        interim_data_path=interim_data_path,
        features_path=features_path).set_display_name('Read raw data')
    build_features_task = build_features(
        features_path=features_path,
        processed_data_path=processed_data_path,
        interim_features=interim_features_import.output,
        interim_data=read_raw_data_task.outputs['interim_data']
    ).set_display_name('Build features')
    split_data_task = split_data(
        train_data_size=train_data_size,
        processed_data=build_features_task.outputs['processed_data']
    ).set_display_name('Split data')
    forecast_features_import = import_forecast_features(
        features_path=features_path).set_display_name(
            'Import forecast features')
    with ParallelFor(forecast_features_import.output) as feature:
        train_task = train(
            project_id=project_id,
            region=region,
            timestamp=timestamp,
            feature=feature,
            lookback=lookback,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            train_data_size=train_data_size,
            pipelines_path=pipelines_path,
            train_data=split_data_task.outputs['train_data']).set_display_name(
                'Train')
        evaluate_task = evaluate(
            project_id=project_id,
            region=region,
            feature=feature,
            lookback=lookback,
            batch_size=batch_size,
            timestamp=timestamp,
            test_data=split_data_task.outputs['test_data'],
            scaler_model=train_task.outputs['scaler_model'],
            keras_model=train_task.outputs['keras_model']).set_display_name(
                'Evaluate')
        import_champion_metrics_task = import_champion_metrics(
            feature=feature).set_display_name('Import champion metrics')
        compare_task = compare_models(
            challenger_metrics=evaluate_task.outputs['eval_metrics'],
            champion_metrics=import_champion_metrics_task.
            outputs['champion_metrics']).set_display_name('Compare metrics')
        with Condition(compare_task.output == 'true', name='chall better'):
            upload_model_to_registry(
                project_id=project_id,
                region=region,
                feature=feature,
                models_path=models_path,
                deploy_image=deploy_image,
                parameters=train_task.outputs['parameters'],
                scaler_model=train_task.outputs['scaler_model'],
                keras_model=train_task.outputs['keras_model'],
                metrics=evaluate_task.outputs['eval_metrics']
            ).set_display_name('Upload to registry')


def get_args() -> argparse.Namespace:
    """
    The get_args function parses the arguments passed to the script.
    It returns a Namespace object containing all of the arguments
      
    Returns:
        An argparse
    """
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--compile_only', action='store_true')
    parser.add_argument('--enable_caching', action='store_true')
    parser.add_argument('--timestamp',
                        type=str,
                        default=datetime.now().strftime('%Y%m%d%H%M%S'))
    # Training
    parser.add_argument('--lookback', type=int, default=120)
    parser.add_argument('--train_data_size', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=50)
    # # Hyper parameters
    parser.add_argument('--lstm_units', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    return parser.parse_args()


def get_envs() -> dict:
    """
    The get_envs function returns a dictionary of environment variables that are 
    required by the pipeline.
    
    Returns:
        A dictionary of the environment variables that are needed to run this pipeline
    """
    return {
        'project_id': os.environ['PROJECT_ID'],
        'region': os.environ['REGION'],
        'pipelines_path': os.environ['PIPELINES_PATH'],
        'raw_data_path': os.environ['RAW_DATA_PATH'],
        'interim_data_path': os.environ['INTERIM_DATA_PATH'],
        'processed_data_path': os.environ['PROCESSED_DATA_PATH'],
        'models_path': os.environ['MODELS_PATH'],
        'features_path': os.environ['FEATURES_PATH'],
        'deploy_image': os.environ['DEPLOY_IMAGE']
    }


if __name__ == '__main__':
    args = get_args()
    envs = get_envs()
    pipelines_uri = os.environ['PIPELINES_URI']
    logging.basicConfig(stream=sys.stdout)
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=f'./compiled/training_{args.timestamp}.json')
    if not args.compile_only:
        training_parameters = {
            'lookback': args.lookback,
            'train_data_size': args.train_data_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patience': args.patience
        }
        hyperparameters = {
            'lstm_units': args.lstm_units,
            'learning_rate': args.learning_rate
        }