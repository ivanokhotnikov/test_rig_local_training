import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from tools import StandardScaler


class TestRigData(Dataset):

    def __init__(self, seq_len, pred_len, scale, flag, features, val_split,
                 test_split, target, **kwargs):
        assert flag in ['train', 'test', 'val']
        assert features in ['M', 'MS', 'S']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag = flag
        self.set_type = type_map[self.flag]
        self.interim_data_path = os.environ['INTERIM_DATA_PATH']
        self.features_path = os.environ['FEATURES_PATH']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.val_split = val_split
        self.test_split = test_split
        self.target = target
        self.features = features
        self.scale = scale
        self._read_interim_data()

    def __len__(self) -> int:
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        seq_x = self.x[x_begin:x_end]
        seq_y = self.y[y_begin:y_end]
        return seq_x, seq_y

    def _read_interim_data(self) -> pd.DataFrame:
        df = pd.read_parquet(
            os.path.join(self.interim_data_path, 'interim.parquet'))
        border1s = [
            0,
            int(len(df) * (1 - self.val_split - self.test_split)),
            int(len(df) * (1 - self.test_split)),
        ]
        border2s = [
            int(len(df) * (1 - self.val_split - self.test_split)),
            int(len(df) * (1 - self.test_split)),
            len(df),
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self._load_forecast_features()
        if self.features == 'M':
            self.target = self.forecast_features
            self.dataset = df.loc[border1:border2, self.forecast_features]
            self.y = self.dataset.values
        else:
            assert self.target in self.forecast_features
            if self.features == 'MS':
                self.dataset = df.loc[border1:border2, self.forecast_features]
                self.y = df.loc[border1:border2,
                                self.target].values.reshape(-1, 1)
            elif self.features == 'S':
                self.dataset = df.loc[border1:border2, self.target]
                self.y = self.dataset.values.reshape(-1, 1)
        if self.features == 'S':
            self.x = self.dataset.values.reshape(-1, 1)
        else:
            self.x = self.dataset.values

        if self.scale:
            self.scaler = StandardScaler()
            train_data = self.x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            self.x = self.scaler.transform(self.x)

    def _load_forecast_features(self):
        with open(os.path.join(self.features_path, 'forecast_features.json'),
                  'r') as fp:
            self.forecast_features = json.load(fp)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TestRigDataPredict(Dataset):

    def __init__(self, root_path, data_path, seq_len, pred_len, flag, features,
                 target, scale, **kwargs):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.flag = flag
        self.target = target
        self.features = features
        self.features_path = os.environ['FEATURES_PATH']
        self._read_train_data()
        self._read_new_data()

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.seq_len
        seq_x = self.x[x_begin:x_end]
        return seq_x

    def _read_train_data(self):
        self.train_data = TestRigData(seq_len=self.seq_len,
                                      pred_len=self.pred_len,
                                      features=self.features,
                                      target=self.target,
                                      scale=True,
                                      flag='train',
                                      val_split=.2,
                                      test_split=.2)
        self.scaler = self.train_data.scaler

    def _read_new_data(self):
        self.df = pd.read_csv(os.path.join(self.root_path, self.data_path),
                              index_col=False,
                              header=0)
        self._load_features()
        self._build_features()
        if self.features == 'M':
            self.target = self.forecast_features
            self.dataset = self.df[self.forecast_features]
            self.y = self.dataset.values
        else:
            assert self.target in self.forecast_features
            if self.features == 'MS':
                self.dataset = self.df[self.forecast_features]
                self.y = self.df[self.target].values.reshape(-1, 1)
            elif self.features == 'S':
                self.dataset = self.df[self.target]
                self.y = self.dataset.values.reshape(-1, 1)
        if self.features == 'S':
            self.x = self.dataset.values.reshape(-1, 1)
        else:
            self.x = self.dataset.values

        if self.scale:
            self.x = self.scaler.transform(self.x)

    def _load_features(self):
        with open(os.path.join(self.features_path, 'interim_features.json'),
                  'r') as features_file:
            self.interim_features_list = list(json.loads(features_file.read()))
        self.no_time_features = [
            f for f in self.interim_features_list
            if f not in ('TIME', ' DATE', 'DATE')
        ]
        with open(os.path.join(self.features_path, 'forecast_features.json'),
                  'r') as fp:
            self.forecast_features = json.load(fp)

    def _build_features(self):
        self.df[self.no_time_features] = self.df[self.no_time_features].apply(
            pd.to_numeric, errors='coerce', downcast='float')
        self.df = self.df.drop(self.df[self.df['STEP'] == 0].index,
                               axis=0).reset_index(drop=True)
        self.df.dropna(axis=0, inplace=True)
        self.df['DRIVE_POWER'] = (self.df['M1 SPEED'] * self.df['M1 TORQUE'] *
                                  torch.pi / 30 / 1e3).astype(float)
        self.df['LOAD_POWER'] = abs(self.df['D1 RPM'] * self.df['D1 TORQUE'] *
                                    torch.pi / 30 / 1e3).astype(float)
        self.df['CHARGE_MECH_POWER'] = (self.df['M2 RPM'] *
                                        self.df['M2 Torque'] * torch.pi / 30 /
                                        1e3).astype(float)
        self.df['CHARGE_HYD_POWER'] = (self.df['CHARGE PT'] * 1e5 *
                                       self.df['CHARGE FLOW'] * 1e-3 / 60 /
                                       1e3).astype(float)
        self.df['SERVO_MECH_POWER'] = (self.df['M3 RPM'] *
                                       self.df['M3 Torque'] * torch.pi / 30 /
                                       1e3).astype(float)
        self.df['SERVO_HYD_POWER'] = (self.df['Servo PT'] * 1e5 *
                                      self.df['SERVO FLOW'] * 1e-3 / 60 /
                                      1e3).astype(float)
        self.df['SCAVENGE_POWER'] = (self.df['M5 RPM'] * self.df['M5 Torque'] *
                                     torch.pi / 30 / 1e3).astype(float)
        self.df['MAIN_COOLER_POWER'] = (self.df['M6 RPM'] *
                                        self.df['M6 Torque'] * torch.pi / 30 /
                                        1e3).astype(float)
        self.df['GEARBOX_COOLER_POWER'] = (self.df['M7 RPM'] *
                                           self.df['M7 Torque'] * torch.pi /
                                           30 / 1e3).astype(float)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
