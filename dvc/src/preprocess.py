import argparse
import json
import logging
import os
from datetime import datetime
from math import pi

import pandas as pd

logging.basicConfig(level=logging.INFO)


def preprocess(interim_features_file: str, all_raw_data_file: str,
               interim_data_file: str) -> None:
    """
    The preprocess function reads the interim features file and all raw data
    file, then creates a new DataFrame with only the columns specified in the
    interim features file. The function also drops rows containing NA values and
    rows where STEP is equal to 0. It downcasts floats to float32, which saves on
    memory usage when training models later. Finally, it adds some time-based
    features such as RUNNING_SECONDS (the number of seconds since start of run)
    and RUNNING_HOURS (RUNNING_SECONDS converted to hours) and power-based features.
    The interim data is then saved.

    Args:
        interim_features_file: str: Specify the path to a json
        all_raw_data_file: str: Specify the path to the all raw data
        interim_data_file: str: Specify the path to the interim data

    Returns:
        None
    """
    start = datetime.now()
    with open(interim_features_file, 'r') as fp:
        interim_features = json.load(fp)
    logging.info('Interim features file was read')
    interim_df = pd.read_csv(all_raw_data_file,
                             usecols=interim_features,
                             header=0,
                             low_memory=False,
                             index_col=False)
    logging.info('All raw data file was read')
    interim_df = interim_df.apply(pd.to_numeric,
                                  downcast='float',
                                  errors='coerce')
    logging.info('Downcasted')
    interim_df.dropna(axis=0, inplace=True)
    logging.info('NAs rows droped')
    interim_df = interim_df.drop(interim_df[interim_df['STEP'] == 0].index,
                                 axis=0).reset_index(drop=True)
    interim_df.drop(columns=['STEP'], inplace=True, errors='ignore')
    logging.info('Step zero removed')
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
    logging.info('Power features added')
    interim_df['RUNNING_SECONDS'] = (pd.to_timedelta(
        range(len(interim_df)), unit='s').total_seconds()).astype(int)
    interim_df['RUNNING_HOURS'] = (interim_df['RUNNING_SECONDS'] /
                                   3600).astype(float)
    logging.info('Time features added')
    interim_df.columns = interim_df.columns.str.strip()
    interim_df.columns = interim_df.columns.str.replace(' ', '_')
    interim_df.columns = interim_df.columns.str.upper()
    os.makedirs('/'.join(interim_data_file.split('/')[:-1]), exist_ok=True)
    interim_df.to_parquet(interim_data_file, index=False)
    logging.info('Interim dataframe saved')
    forecast_features = [
        f for f in interim_df.columns
        if any((('POWER' in f), ('VIBRATION' in f)))
    ]
    forecast_features_file = ('/').join(
        interim_features_file.split('/')[:-1] + ['forecast_features.json'])
    with open(forecast_features_file, 'w') as fp:
        json.dump(forecast_features, fp)
    logging.info('Forecast features saved')
    logging.info(
        f'Elapsed time {os.path.basename(__file__)} {datetime.now() - start}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim_features_file', type=str)
    parser.add_argument('--all_raw_data_file', type=str)
    parser.add_argument('--interim_data_file', type=str)
    args = parser.parse_args()
    preprocess(interim_features_file=args.interim_features_file,
               all_raw_data_file=args.all_raw_data_file,
               interim_data_file=args.interim_data_file)
