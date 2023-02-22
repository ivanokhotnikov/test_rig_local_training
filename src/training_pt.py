import json
import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset


class TestRigDataset(Dataset):

    def __init__(self,
                 lookback: int,
                 scale: bool = True,
                 flag: str = 'train',
                 target: str | None = None,
                 features: str = 'S',
                 val_split: float = .2,
                 test_split: float = .2) -> None:
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.interim_data_path = './data/interim'
        self.features_path = './conf'
        self.lookback = lookback
        self.val_split = val_split
        self.test_split = test_split
        self.target = target
        self.features = features
        self.scale = scale
        self.__read_data__()

    def __len__(self) -> int:
        return len(self.dataset)

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

        if self.features == 'S' or self.features == 'MS':
            if self.target:
                self.dataset = interim_df.loc[border1:border2, [self.target]]
            else:
                raise AttributeError('Specify target')
        else:
            self.target = self.forecast_features
        if 'M' in self.features:
            self.dataset = interim_df.loc[border1:border2,
                                          self.forecast_features]
        else:
            self.dataset = interim_df.loc[border1:border2, self.target]

    def __getitem__(self, index: int):
        seq_x = self.dataset[index:index + self.lookback].values
        seq_y = self.dataset[index + self.lookback:index + self.lookback +
                             1].values
        return seq_x, seq_y
