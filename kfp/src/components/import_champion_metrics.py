from kfp.v2.dsl import Metrics, Output, component
from utils.dependencies import PYTHON310


@component(base_image=PYTHON310)
def import_champion_metrics(feature: str,
                            champion_metrics: Output[Metrics]) -> None:
    """
    The import_champion_metrics function imports the champion model's metrics from a JSON file
    and adds them to the champion model's metadata. The function takes two arguments:
        - feature: A string indicating which feature is being modeled.
        - champion_metrics: An Output object representing the pipeline's metrics for this feature.

    Args:
        feature: str: Specify the feature that is being forecasted
        champion_metrics: Output[Metrics]: Store the champion model's metrics
    """
    import json
    import os

    with open(os.path.join('gcs', 'models_forecasting', f'{feature}.json'),
              'r') as registry_metrics_file:
        champion_metrics_dict = json.load(registry_metrics_file)
    with open(champion_metrics.path + '.json', 'w') as pipeline_metrics_file:
        json.dump(champion_metrics_dict, pipeline_metrics_file)
    champion_metrics.metadata['feature'] = feature
