from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component
from utils.dependencies import (GOOGLE_CLOUD_AIPLATFORM, PANDAS, PROTOBUF,
                                PYARROW, SKLEARN, TF_TRAIN_GPU_IMAGE)


@component(base_image=TF_TRAIN_GPU_IMAGE,
           packages_to_install=[
               PANDAS, SKLEARN, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF, PYARROW
           ])
def evaluate(project_id: str, region: str, feature: str, lookback: int,
             batch_size: int, timestamp: str, test_data: Input[Dataset],
             scaler_model: Input[Model], keras_model: Input[Model],
             eval_metrics: Output[Metrics]) -> None:
    """
    The evaluate function loads the test data, creates a scaler model and a keras model from the training data, and evaluates them against the test data.

    Args:
        project_id: str: Specify the project id where the ai platform training and prediction resources will be created
        region: str: Specify the region in which to run the training job
        feature: str: Identify the feature being evaluated
        lookback: int: Specify how many previous time steps to use as input variables to predict the next time period
        batch_size: int: Specify the number of samples to work on before updating the model
        timestamp: str: Tag the evaluation metrics file with a timestamp
        test_data: Input[Dataset]: Pass the test data to the evaluate function
        scaler_model: Input[Model]: Save the scaler model to gcs
        keras_model: Input[Model]: Pass the trained model to the evaluate function
        eval_metrics: Output[Metrics]: Log the metrics to ai platform
    """
    import json

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    aip.init(project=project_id, location=region, experiment=timestamp)
    aip.start_run(run=feature.lower().replace('_', '-'), resume=True)
    test_df = pd.read_parquet(test_data.path + '.parquet')
    test_data = test_df[feature].values.reshape(-1, 1)
    scaler = joblib.load(scaler_model.path + '.joblib')
    scaled_test = scaler.transform(test_data)
    x_test, y_test = [], []
    for i in range(lookback, len(scaled_test)):
        x_test.append(scaled_test[i - lookback:i])
        y_test.append(scaled_test[i])
    x_test = np.stack(x_test)
    y_test = np.stack(y_test)
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    results = forecaster.evaluate(x_test,
                                  y_test,
                                  verbose=1,
                                  batch_size=batch_size,
                                  return_dict=True)
    aip.log_metrics(results)
    results['timestamp'] = timestamp
    with open(eval_metrics.path + '.json', 'w') as metrics_file:
        metrics_file.write(json.dumps(results))
    for k, v in results.items():
        eval_metrics.log_metric(k, v)
    eval_metrics.metadata['feature'] = feature
    aip.end_run()
