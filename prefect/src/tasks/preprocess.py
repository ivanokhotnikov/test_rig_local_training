import json
import os
from math import pi

import pandas as pd
from prefect.tasks import task_input_hash

from prefect import task


@task(cache_key_fn=task_input_hash, refresh_cache=False)
def preprocess(df: pd.DataFrame, features_path: str,
               interim_data_path: str) -> tuple:
    """
    The preprocess function takes a dataframe and returns two items:
    - A list of features to be used for forecasting.
    - A preprocessed dataframe with the selected features.


    Args:
        df: pd.DataFrame: Pass the dataframe to be processed
        features_path: str: Specify the path to the features file
        interim_data_path: str: Specify the path to save the interim dataframe

    Returns:
        A tuple with two elements: features (list of str), interim dataframe (pandas DataFrame)
    """
    with open(os.path.join(features_path, 'interim_features.json'), 'r') as fp:
        interim_features = json.load(fp)
    interim_df = df[interim_features].copy(deep=True)
    interim_df = interim_df.apply(pd.to_numeric,
                                  downcast='float',
                                  errors='coerce')
    interim_df.dropna(axis=0, inplace=True)
    interim_df = interim_df.drop(interim_df[interim_df['STEP'] == 0].index,
                                 axis=0).reset_index(drop=True)
    interim_df.drop(columns=['STEP'], inplace=True, errors='ignore')

    interim_df['DRIVE_POWER'] = (interim_df['M1 SPEED'] *
                                 interim_df['M1 TORQUE'] * pi / 30 /
                                 1e3).astype(float)
    interim_df['LOAD_POWER'] = abs(interim_df['D1 RPM'] *
                                   interim_df['D1 TORQUE'] * pi / 30 /
                                   1e3).astype(float)
    interim_df['CHARGE_MECH_POWER'] = (interim_df['M2 RPM'] *
                                       interim_df['M2 Torque'] * pi / 30 /
                                       1e3).astype(float)
    interim_df['CHARGE_HYD_POWER'] = (interim_df['CHARGE PT'] * 1e5 *
                                      interim_df['CHARGE FLOW'] * 1e-3 / 60 /
                                      1e3).astype(float)
    interim_df['SERVO_MECH_POWER'] = (interim_df['M3 RPM'] *
                                      interim_df['M3 Torque'] * pi / 30 /
                                      1e3).astype(float)
    interim_df['SERVO_HYD_POWER'] = (interim_df['Servo PT'] * 1e5 *
                                     interim_df['SERVO FLOW'] * 1e-3 / 60 /
                                     1e3).astype(float)
    interim_df['SCAVENGE_POWER'] = (interim_df['M5 RPM'] *
                                    interim_df['M5 Torque'] * pi / 30 /
                                    1e3).astype(float)
    interim_df['MAIN_COOLER_POWER'] = (interim_df['M6 RPM'] *
                                       interim_df['M6 Torque'] * pi / 30 /
                                       1e3).astype(float)
    interim_df['GEARBOX_COOLER_POWER'] = (interim_df['M7 RPM'] *
                                          interim_df['M7 Torque'] * pi / 30 /
                                          1e3).astype(float)
    interim_df['RUNNING_SECONDS'] = (pd.to_timedelta(
        range(len(interim_df)), unit='s').total_seconds()).astype(int)
    interim_df['RUNNING_HOURS'] = (interim_df['RUNNING_SECONDS'] /
                                   3600).astype(float)
    interim_df.columns = interim_df.columns.str.strip()
    interim_df.columns = interim_df.columns.str.replace(' ', '_')
    interim_df.columns = interim_df.columns.str.upper()
    interim_df.to_parquet(os.path.join(interim_data_path, 'interim.parquet'),
                          index=False)
    forecast_features = [
        f for f in interim_df.columns
        if any((('POWER' in f), ('VIBRATION' in f)))
    ]
    with open(os.path.join(features_path, 'forecast_features.json'),
              'w') as fp:
        json.dump(forecast_features, fp)
    return forecast_features, interim_df
