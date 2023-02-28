import json
import os

import pandas as pd
from torch.utils.data import Dataset

from tools import StandardScaler


class TestRigData(Dataset):

    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 scale: bool,
                 flag: str,
                 features: str,
                 val_split: float,
                 test_split: float,
                 target: str | None = None) -> None:
        assert flag in ['train', 'test', 'val']
        assert features in ['M', 'MS', 'S']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.interim_data_path = './data/interim'
        self.features_path = './conf'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.val_split = val_split
        self.test_split = test_split
        self.target = target
        self.features = features
        self.scale = scale
        self.__read_data__()

    def __len__(self) -> int:
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __read_data__(self) -> pd.DataFrame:
        interim_df = pd.read_parquet(
            os.path.join(self.interim_data_path, 'interim.parquet'))
        size = len(interim_df)
        border1s = [
            0,
            int(size * (1 - self.val_split - self.test_split)),
            int(size * (1 - self.test_split))
        ]
        border2s = [
            int(size * (1 - self.val_split - self.test_split)),
            int(size * (1 - self.test_split)), size
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        with open(os.path.join(self.features_path, 'forecast_features.json'),
                  'r') as fp:
            self.forecast_features = json.load(fp)
        if self.features == 'M':
            self.target = self.forecast_features
            self.dataset = interim_df.loc[border1:border2,
                                          self.forecast_features]
            self.y = self.dataset.values
        else:
            assert self.target in self.forecast_features
            if self.features == 'MS':
                self.dataset = interim_df.loc[border1:border2,
                                              self.forecast_features]
                self.y = interim_df.loc[border1:border2, self.target].values
            elif self.features == 'S':
                self.dataset = interim_df.loc[border1:border2, self.target]
                self.y = self.dataset.values
        self.x = self.dataset.values

        if self.scale:
            self.scaler = StandardScaler()
            train_data = self.x[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            self.x = self.scaler.transform(self.x)

    def __getitem__(self, index: int):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        seq_x = self.x[x_begin:x_end]
        seq_y = self.y[y_begin:y_end]
        return seq_x, seq_y
