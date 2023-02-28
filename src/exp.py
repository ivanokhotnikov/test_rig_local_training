import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import TestRigData
from models import LSTM


class Experiment:

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(
                self.args.gpu
            ) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _select_criterion(self):
        return nn.MSELoss()

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _get_data(self, flag):
        if flag in ['test', 'pred']:
            shuffle_flag = False
        else:
            shuffle_flag = True
        data = TestRigData(seq_len=self.args.seq_len,
                           pred_len=self.args.pred_len,
                           features=self.args.features,
                           val_split=self.args.val_split,
                           test_split=self.args.test_split,
                           target=self.args.target,
                           scale=True,
                           flag=flag)
        loader = DataLoader(dataset=data,
                            batch_size=self.args.batch_size,
                            shuffle=shuffle_flag,
                            drop_last=True)
        return data, loader

    def _build_model(self):
        model = LSTM
        return model(input_size=self.train_data.x.shape[-1],
                     output_size=self.train_data.y.shape[-1],
                     hidden_size=self.args.hidden_size,
                     num_layers=self.args.num_layers,
                     batch_size=self.args.batch_size).double()

    def train(self):
        self.train_data, self.train_loader = self._get_data(flag='train')
        if self.args.features == 'S':
            self.train_data.x = self.train_data.x.reshape(-1, 1)
            self.train_data.y = self.train_data.y.reshape(-1, 1)
        elif self.args.features == 'MS':
            self.train_data.y = self.train_data.y.reshape(-1, 1)
        self.model = self._build_model().to(self.device)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        for epoch in range(self.args.epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x.double().to(self.device)
                batch_y.double().to(self.device)
                model_optim.zero_grad()
                pred = self.model(batch_x.double())
                pred = pred[:,
                            -self.args.pred_len:, :].double().to(self.device)
                rmse_loss = torch.sqrt(criterion(pred, batch_y))
                train_loss.append(rmse_loss.item())
                if i % self.args.log_interval == 0:
                    print('\titer: {0}, epoch: {1} | loss: {2:.3f}'.format(
                        i, epoch, rmse_loss.item()))
                rmse_loss.backward()
                model_optim.step()
                if self.args.dry_run: break
            if self.args.dry_run: break

    def validate(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass
