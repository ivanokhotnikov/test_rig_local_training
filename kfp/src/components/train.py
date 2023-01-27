from kfp.v2.dsl import (Artifact, Dataset, Input, Metrics, Model, Output,
                        component)

from utils.dependencies import (GOOGLE_CLOUD_AIPLATFORM, PANDAS, PROTOBUF,
                                PYARROW, SKLEARN, TF_TRAIN_GPU_IMAGE)


@component(base_image=TF_TRAIN_GPU_IMAGE,
           packages_to_install=[
               PANDAS, SKLEARN, PROTOBUF, GOOGLE_CLOUD_AIPLATFORM, PYARROW
           ])
def train(project_id: str, region: str, feature: str, lookback: int,
          lstm_units: int, learning_rate: float, epochs: int, batch_size: int,
          patience: int, timestamp: str, train_data_size: float,
          pipelines_path: str, train_data: Input[Dataset],
          scaler_model: Output[Model], keras_model: Output[Model],
          train_metrics: Output[Metrics],
          parameters: Output[Artifact]) -> None:
    """
    The train function trains a model to predict the next value in a time series.
    
    Args:
        project_id: str: Specify the project id where the ai platform training and prediction resources will be created
        region: str: Specify the region in which to run the training job
        feature: str: Identify the model
        lookback: int: Determine how many pr2evious time steps to use as input variables for the model
        lstm_units: int: Define the number of units in the lstm layer
        learning_rate: float: Control the step size in updating the weights
        epochs: int: Specify the number of epochs to train for
        batch_size: int: Determine the number of samples in each batch
        patience: int: Determine how many epochs to wait before stopping training if the loss hasn't improved
        timestamp: str: Create a unique folder for this run
        train_data_size: float: Determine the size of the training data
        pipelines_path: str: Store the model in a specific location
        train_data: Input[Dataset]: Pass the training dataset to the train function
        scaler_model: Output[Model]: Save the scaler model
        keras_model: Output[Model]: Save the model
        train_metrics: Output[Metrics]: Log the metrics of the training job
        parameters: Output[Artifact]: Save the parameters used in training
    """
    import json
    import os

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    train_df = pd.read_parquet(train_data.path + '.parquet')
    train_data = train_df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaler_model.metadata['feature'] = feature
    joblib.dump(scaler, scaler_model.path + '.joblib')
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_train_data)):
        x_train.append(scaled_train_data[i - lookback:i])
        y_train.append(scaled_train_data[i])
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    forecaster = keras.models.Sequential(name=f'{feature}_forecaster')
    forecaster.add(
        keras.layers.LSTM(lstm_units,
                          input_shape=(x_train.shape[1], x_train.shape[2]),
                          return_sequences=False))
    forecaster.add(keras.layers.Dense(1))
    forecaster.compile(
        loss=keras.losses.mean_squared_error,
        metrics=keras.metrics.RootMeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
    history = forecaster.fit(
        x_train,
        y_train,
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=patience,
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.75,
                                              patience=patience // 2,
                                              verbose=1,
                                              mode='min'),
            keras.callbacks.TensorBoard(log_dir=os.path.join(
                pipelines_path, 'tensorboards', feature, timestamp),
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=True,
                                        update_freq='epoch')
        ])
    for k, v in history.history.items():
        history.history[k] = [float(vi) for vi in v]
        train_metrics.log_metric(k, history.history[k])
    train_metrics.metadata['feature'] = feature
    keras_model.metadata['feature'] = feature
    forecaster.save(keras_model.path + '.h5')
    params = {
        'lookback': lookback,
        'lstm_units': lstm_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'train_data_size': train_data_size,
        **forecaster.get_config(),
        **forecaster.optimizer.get_config(),
        **forecaster.history.params
    }
    aip.init(experiment=timestamp, project=project_id, location=region)
    aip.start_run(run=feature.lower().replace('_', '-'))
    aip.log_params({
        'lookback': lookback,
        'lstm_units': lstm_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'train_data_size': train_data_size
    })
    for k in params:
        if type(params[k]) in (np.float16, np.float32, np.float64):
            params[k] = float(params[k])
    with open(parameters.path + '.json', 'w') as params_file:
        json.dump(params, params_file)