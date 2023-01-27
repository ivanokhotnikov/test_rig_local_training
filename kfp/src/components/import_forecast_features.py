from kfp.v2.dsl import component
from utils.dependencies import PYTHON310


@component(base_image=PYTHON310)
def import_forecast_features(features_path: str) -> list:
    """
    The import_forecast_features function imports the forecast features from a JSON file.
    The function returns a string representation of the imported forecast features.

    Args:
        features_path: str: Specify the path to the folder containing the forecast_features

    Returns:
        A string of the forecast features
    """
    import json
    import os
    with open(os.path.join(features_path, 'forecast_features.json'),
              'r') as final_features_file:
        forecast_features = json.loads(final_features_file.read())
    return forecast_features
