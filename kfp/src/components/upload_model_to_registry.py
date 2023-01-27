from kfp.v2.dsl import Artifact, Input, Metrics, Model, component

from utils.dependencies import (GOOGLE_CLOUD_AIPLATFORM, PROTOBUF, SKLEARN,
                                TF_TRAIN_CPU_IMAGE)


@component(base_image=TF_TRAIN_CPU_IMAGE,
           packages_to_install=[SKLEARN, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF])
def upload_model_to_registry(project_id: str, region: str, feature: str,
                             deploy_image: str, models_path: str,
                             parameters: Input[Artifact],
                             scaler_model: Input[Model],
                             keras_model: Input[Model],
                             metrics: Input[Metrics]) -> None:
    """
    The upload_model_to_registry function uploads the model, its parameters and metrics to the AI Platform Model Registry.
    
    Args:
        project_id: str: Specify the project id in which to create the model
        region: str: Specify the region where the model will be deployed
        feature: str: Name the model
        deploy_image: str: Specify the container image that will be used to deploy the model
        models_path: str: Specify the path to the models folder
        parameters: Input[Artifact]: Pass the path to the model artifacts
        scaler_model: Input[Model]: Upload the scaler model to ai platform
        keras_model: Input[Model]: Upload the model to ai platform
        metrics: Input[Metrics]: Pass the metrics from the training pipeline to the model registry
    """
    import json
    import os

    import google.cloud.aiplatform as aip
    import joblib
    from tensorflow import keras

    scaler = joblib.load(scaler_model.path + '.joblib')
    joblib.dump(scaler, os.path.join(models_path, f'{feature}.joblib'))
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    forecaster.save(os.path.join(models_path, f'{feature}.h5'))
    with open(metrics.path + '.json', 'r') as pipeline_metrics_file:
        eval_metrics_dict = json.load(pipeline_metrics_file)
    with open(os.path.join(models_path, f'{feature}.json'),
              'w') as registry_metrics_file:
        registry_metrics_file.write(json.dumps(eval_metrics_dict))
    with open(parameters.path + '.json', 'r') as fp:
        params = json.load(fp)
    with open(os.path.join(models_path, f'{feature}_params.json'), 'w') as fp:
        json.dump(params, fp)
    forecaster.save(os.path.join(models_path, 'registry', feature))
    models = [
        model.display_name
        for model in aip.Model.list(project=project_id, location=region)
    ]
    aip.init(project=project_id, location=region)
    if feature not in models:
        model = aip.Model.upload(project=project_id,
                                 location=region,
                                 display_name=feature,
                                 artifact_uri=os.path.join(
                                     models_path, 'registry', feature),
                                 serving_container_image_uri=deploy_image,
                                 is_default_version=True)
    else:
        for model in aip.Model.list(project=project_id, location=region):
            if model.display_name == feature:
                model = aip.Model.upload(
                    project=project_id,
                    location=region,
                    parent_model=model.name,
                    display_name=feature,
                    artifact_uri=os.path.join(models_path, 'registry',
                                              feature),
                    serving_container_image_uri=deploy_image,
                    is_default_version=True)
                break
