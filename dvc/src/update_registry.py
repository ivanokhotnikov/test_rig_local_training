import argparse
import json
import logging
import os
from datetime import datetime

from keras.models import load_model

logging.basicConfig(level=logging.INFO)

def update_registry(champion_metrics_path: str, challenger_metrics_path: str,
                    target: str) -> None:
    """
    The update_registry function updates the champion model in the registry with a new challenger.
    If there is no champion, then it uploads the challenger to the registry.
        
    Args:
        champion_metrics_path: str: Specify the path to the file containing the champion model's metrics
        challenger_metrics_path: str: Specify the path to the challenger metrics
        target: str: Specify the model that is being updated
    
    Returns:
        None
    """
    start = datetime.now()
    with open(challenger_metrics_path, 'r') as fp:
        challenger_metrics = json.load(fp)
    if os.path.exists(champion_metrics_path):
        with open(champion_metrics_path, 'r') as fp:
            champion_metrics = json.load(fp)
        if challenger_metrics['test_loss'] < champion_metrics['test_loss']:
            upload_challenger(target, challenger_metrics)
    else:
        upload_challenger(target, challenger_metrics)
    logging.info(
        f'Elapsed time {os.path.basename(__file__)}@{target} {datetime.now() - start}'
    )


def upload_challenger(target: str, challenger_metrics: dict) -> None:
    """
    The upload_challenger function uploads the challenger model to the registry.
    
    Args:
        target: str: Specify the target column for which we want to upload a challenger model
        challenger_metrics: dict: Store the metrics of the challenger model
    
    Returns:
        None
    """
    model = load_model(f'./artifacts/temp/model_{target}.h5')
    model.save(f'./artifacts/registry/model_{target}.h5')
    with open(f'./artifacts/temp/val_metrics_{target}.json', 'r') as fp:
        val_metrics = json.load(fp)
    with open(f'./artifacts/registry/val_metrics_{target}.json', 'w') as fp:
        json.dump(val_metrics, fp)
    with open(f'./artifacts/registry/test_metrics_{target}.json', 'w') as fp:
        json.dump(challenger_metrics, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--champion', type=str)
    parser.add_argument('--challenger', type=str)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()
    update_registry(champion_metrics_path=args.champion,
                    challenger_metrics_path=args.challenger,
                    target=args.target)
