import argparse
import logging
import os
import sys
import time

from google.api_core.exceptions import NotFound
from google.cloud import aiplatform, storage


def vertex(custom_jobs: bool, pipeline_jobs: bool, artifacts: bool,
           models: bool, tensorboards: bool, experiments: bool,
           metadata_store: bool, tensorboard_events: bool, all: bool,
           sleep: float, project_id: str, region: str, project_number: str,
           pipelines_uri: str, **kwargs):
    """
    The vertex function cleans up all AI Platform resources in a given region.
    
    Args:
        custom_jobs: bool: Clean custom jobs
        pipeline_jobs: bool: Clean the pipeline jobs
        artifacts: bool: Clean the artifacts in the region
        models: bool: Determine whether or not to clean the models
        tensorboards: bool: Clean tensorboard events
        experiments: bool: Clean the experiments
        metadata_store: bool: Clean the metadata store
        tensorboard_events: bool: Clean the tensorboard events in the metadata store
        all: bool: Clean all the resources in a region
        sleep: float: Wait for a given number of seconds between each api call
        project_id: str: Specify the project id
        region: str: Specify the region where you want to clean your resources
        project_number: str: Specify the project number in which the resources will be cleaned
        pipelines_uri: str: Specify the gcs bucket where your pipelines are stored
        **kwargs: Catch any additional arguments that may be passed to the function
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.info(f'custom_jobs is set to {custom_jobs}')
    logging.info(f'pipeline_jobs is set to {pipeline_jobs}')
    logging.info(f'artifacts is set to {artifacts}')
    logging.info(f'models is set to {models}')
    logging.info(f'tensorboards is set to {tensorboards}')
    logging.info(f'experiments is set to {experiments}')
    logging.info(f'metadata_store is set to {metadata_store}')
    logging.info(f'tensorboard_events is set to {tensorboard_events}')
    logging.info(f'all is set to {all}')
    if all:
        custom_jobs = True
        pipeline_jobs = True
        artifacts = True
        models = True
        tensorboards = True
        experiments = True
        metadata_store = True
        tensorboard_events = True
    elif not any([
            custom_jobs, pipeline_jobs, artifacts, models, tensorboards,
            experiments, metadata_store, tensorboard_events
    ]):
        logging.info(f'No flags set to clean!')
        sys.exit()
    aiplatform.init(project=project_id, location=region)
    try:
        if custom_jobs:
            for job in aiplatform.CustomJob.list():
                job.delete()
                time.sleep(sleep)
            logging.info(f'Custom jobs have been cleaned in {region}')
        if pipeline_jobs:
            for job in aiplatform.PipelineJob.list():
                job.delete()
                time.sleep(sleep)
            logging.info(f'Custom jobs have been cleaned in {region}')
        if artifacts:
            for art in aiplatform.Artifact.list():
                art.delete()
                time.sleep(sleep)
            logging.info(f'Artifacts have been cleaned in {region}')
        if models:
            for model in aiplatform.Model.list():
                model.delete()
                time.sleep(sleep)
            logging.info(f'Models have been cleaned in {region}')
        if tensorboards:
            for tb in aiplatform.Tensorboard.list():
                tb.delete()
                time.sleep(sleep)
            logging.info(f'Tensorboards have been cleaned in {region}')
        if experiments:
            for exp in aiplatform.Experiment.list():
                for exp_run in aiplatform.ExperimentRun.list(experiment=exp):
                    exp_run.delete()
                    time.sleep(sleep)
                logging.info(
                    f'Experiment runs in {exp.name} have been cleaned in {region}'
                )
                exp.delete()
                time.sleep(sleep)
            logging.info(f'Experiments have been cleaned in {region}')
        if metadata_store:
            client = storage.Client()
            for blob in client.list_blobs(
                    bucket_or_name=pipelines_uri.lstrip('gs://'),
                    prefix=project_number):
                blob.delete()
            for blob in client.list_blobs(
                    bucket_or_name=pipelines_uri.lstrip('gs://'),
                    prefix='vertex_ai_auto_staging'):
                blob.delete()
            logging.info(f'Metadata store has been cleaned in {region}')
        if tensorboard_events:
            for blob in client.list_blobs(
                    bucket_or_name=pipelines_uri.lstrip('gs://'),
                    prefix='tensorboards'):
                blob.delete()
            logging.info(f'Tensorboard events have been cleaned in {region}')
    except NotFound as NotFoundError:
        logging.info(f'{NotFoundError} in {region}')


def get_args():
    """
    The get_args function parses the arguments passed to the script.
    The arguments are used to determine which objects in GCP will be cleaned up.
    
    Returns:
        A dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_jobs', action='store_true')
    parser.add_argument('--pipeline_jobs', action='store_true')
    parser.add_argument('--artifacts', action='store_true')
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--tensorboards', action='store_true')
    parser.add_argument('--experiments', action='store_true')
    parser.add_argument('--metadata_store', action='store_true')
    parser.add_argument('--tensorboard_events', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sleep', type=float, default=1.05)
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
        'project_number': os.environ['PROJECT_NUMBER'],
        'pipelines_uri': os.environ['PIPELINES_URI'],
    }


if __name__ == '__main__':
    args = get_args()
    envs = get_envs()
    vertex(**vars(args), **envs)
