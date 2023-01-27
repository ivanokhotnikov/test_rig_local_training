# Test rig forecasting local training

The repository contains the source code for implementations of the local training on the test rig data. The implementations differ mainly in the orchestration and tracking stacks. The implemented orchestrators include [dvc](dvc), [prefect](prefect) and [kfp](kfp). All aspire to leverage `mlflow` for experiment tracking and `optuna` for tuning.

## Set up environment

```
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools
pip install -r conf/requirements.txt 
```

## DVC orchestrated training

```
cd dvc/
```
### Reproduce pipeline
```
dvc repro [OPTIONS]
```
See [reference](https://dvc.org/doc/command-reference/repro) for details on the available `dvc repro` CLI options.
### Run experiment
```
dvc exp run [OPTIONS] [PARAMS]
```
See [reference](https://dvc.org/doc/command-reference/exp/run) for details on the available `dvc exp run` CLI options. The training parameters can be set with `-S` flag. To see the list of the available training parameters, see [params.yaml](dvc/params.yaml).

## Prefect orchestrated training
```
cd prefect/
```
### Execute training run
```
python src/training.py [OPTIONS]
OPTIONS:
    --train_split TRAIN_SPLIT
    --lookback LOOKBACK
    --val_split VAL_SPLIT
    --epochs EPOCHS
    --batch_size BATCH_SIZE
    --patience PATIENCE
    --lstm_units LSTM_UNITS
    --learning_rate LEARNING_RATE
    --folds FOLDS
    --verbose VERBOSE
    --univariate
```
### View runs dashboard
```
prefect orion start
```
See localhost:4200
### View experiments dashboard
```
mlflow ui
```
See localhost:5000