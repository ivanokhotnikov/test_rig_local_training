import os

import torch
from torch.utils.data import DataLoader

from data import TestRigDataset


class Experiment:

    def __init__(self, args):
        self.args = args

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(
                self.args.gpu
            ) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        if flag in ['test', 'pred']:
            shuffle_flag = False
        else:
            shuffle_flag = True
        data = TestRigDataset(lookback=self.args.lookback,
                              features=self.args.features,
                              val_split=self.args.val_split,
                              test_split=self.args.test_split,
                              scale=True,
                              flag=flag)
        loader = DataLoader(dataset=data,
                            batch_size=self.args.batch_size,
                            shuffle=shuffle_flag)
        return data, loader

    def _build_model(self):
        pass

    def _select_criterion(self):
        pass

    def _select_optimizer(self):
        pass

    def validate(self):
        pass

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

    def test(self):
        pass

    def predict(self):
        pass
