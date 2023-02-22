import json
import os
import re
from math import pi

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TestRigDataset(Dataset):

    def __init__(self,
                 lookback: int,
                 univariate=True,
                 scale=True,
                 flag='train',
                 target='LOAD_POWER') -> None:
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.raw_data_path = './data/raw'
        self.interim_data_path = './data/interim'
        self.features_path = './conf'
        self.lookback = lookback
        self.target = target
        self.univaraite = univariate
        self.scale = scale
        self.dataset = self.__read_data__()

    def __len__(self) -> int:
        return len(self.dataset)

    def __read_data__(self) -> pd.DataFrame:
        if self.interim_data_path and os.listdir(self.interim_data_path):
            df = pd.read_csv(os.path.join(self.interim_data_path,
                                          'all_raw.csv'),
                             header=0,
                             index_col=False)
        elif self.raw_data_path:
            df = self.__read_raw_data__(
                raw_data_path=self.raw_data_path,
                interim_data_path=self.raw_data_path.replace('raw', 'interim'))
        self.features, df = self.__preprocess__(
            df,
            features_path='./conf',
            interim_data_path=self.interim_data_path)
        if self.univaraite: self.features = self.target
        size = len(self.dataset)

        border1s = [0, int((size * 0.8) * 0.8), int(size * 0.8)]
        border2s = [int((size * 0.8) * 0.8), int(size * 0.8), int(size)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        return df.loc[border1:border2, self.features]

    def __read_raw_data__(self, raw_data_path: str,
                          interim_data_path: str) -> pd.DataFrame:
        """
        The _read_raw_data function reads all the raw data files from a specified directory,
        concatenates them into a single dataframe and saves it to an interim data directory.
        It also returns the concatenated dataframe.

        Args:
            raw_data_path: str: Specify the path to the raw data files
            interim_data_path: str: Specify the path to save the all_raw_data

        Returns:
            A pandas dataframe with all the raw data files read
        """
        all_raw_df = pd.DataFrame()
        units = []
        for file in os.listdir(raw_data_path):
            try:
                if file.endswith('.csv') and 'RAW' in file:
                    file_raw_df = pd.read_csv(os.path.join(
                        raw_data_path, file),
                                              header=0,
                                              index_col=False)
                elif (file.endswith('.xlsx')
                      or file.endswith('.xls')) and 'RAW' in file:
                    file_raw_df = pd.read_excel(os.path.join(
                        raw_data_path, file),
                                                header=0,
                                                index_col=False)
                else:
                    continue
                unit = int(re.split(r'_|-', file.lstrip('-/HYDhyd0'))[0][-4:])
            except:
                continue
            units.append(unit)
            file_raw_df['UNIT'] = unit
            file_raw_df['TEST'] = int(units.count(unit))
            all_raw_df = pd.concat((all_raw_df, file_raw_df),
                                   ignore_index=True)
            del file_raw_df
        all_raw_df.sort_values(by=['UNIT', 'TEST'],
                               inplace=True,
                               ignore_index=True)
        os.makedirs(interim_data_path, exist_ok=True)
        all_raw_df.to_csv(os.path.join(interim_data_path, 'all_raw.csv'),
                          index=False)
        return all_raw_df

    def __preprocess__(self, df: pd.DataFrame, features_path: str,
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
        with open(os.path.join(features_path, 'interim_features.json'),
                  'r') as fp:
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
                                          interim_df['CHARGE FLOW'] * 1e-3 /
                                          60 / 1e3).astype(float)
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
                                              interim_df['M7 Torque'] * pi /
                                              30 / 1e3).astype(float)
        interim_df['RUNNING_SECONDS'] = (pd.to_timedelta(
            range(len(interim_df)), unit='s').total_seconds()).astype(int)
        interim_df['RUNNING_HOURS'] = (interim_df['RUNNING_SECONDS'] /
                                       3600).astype(float)
        interim_df.columns = interim_df.columns.str.strip()
        interim_df.columns = interim_df.columns.str.replace(' ', '_')
        interim_df.columns = interim_df.columns.str.upper()
        interim_df.to_parquet(os.path.join(interim_data_path,
                                           'interim.parquet'),
                              index=False)
        forecast_features = [
            f for f in interim_df.columns
            if any((('POWER' in f), ('VIBRATION' in f)))
        ]
        with open(os.path.join(features_path, 'forecast_features.json'),
                  'w') as fp:
            json.dump(forecast_features, fp)
        return forecast_features, interim_df

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]


if __name__ == '__main__':
    training_data = TestRigDataset(lookback=120)
    training_dataloader = DataLoader(training_data, shuffle=True)
    size = len(training_dataloader.dataset)
    for batch, (X, y) in enumerate(training_dataloader):
        print(batch)
        print(X)
        print(y)
        break
