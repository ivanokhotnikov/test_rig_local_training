import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import TestRigData
from models import LSTM
from tools import EarlyStopping


class Experiment:

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
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
        self.train_data, train_loader = self._get_data(flag='train')
        self.val_data, val_loader = self._get_data(flag='val')
        self.test_data, test_loader = self._get_data(flag='test')
        self.model = self._build_model().to(self.device)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.join('./runs', timestamp)

        for epoch in range(self.args.epochs):
            print('epoch {}'.format(epoch + 1))
            train_losses = []
            self.model.train()
            epoch_time = datetime.now()
            self.writer = SummaryWriter(
                log_dir=os.path.join(path, f'{epoch+1}'))
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x.double().to(self.device)
                batch_y.double().to(self.device)
                model_optim.zero_grad()
                pred = self.model(batch_x)
                pred = pred[:,
                            -self.args.pred_len:, :].double().to(self.device)
                rmse_loss = torch.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar('loss train/iter', rmse_loss, i)
                train_losses.append(rmse_loss.item())
                if i % self.args.log_interval == 0:
                    print('\titer: {0} | loss: {1:.3f}'.format(
                        i, rmse_loss.item()))
                rmse_loss.backward()
                model_optim.step()
                if self.args.dry_run: break
            self.writer.flush()
            train_loss = train_losses[-1]
            if not self.args.train_only:
                val_loss = self.validate(val_loader, criterion)
                test_loss = self.validate(test_loader, criterion)
                print('\tepoch {0} time: {1} s'.format(
                    epoch + 1,
                    datetime.now() - epoch_time))
                print(
                    '\t\ttrain loss: {0:.3f} val loss: {1:.3f} test loss: {2:.3f}'
                    .format(train_loss, val_loss, test_loss))
                early_stopping(val_loss, self.model, path)
            else:
                print('\tepoch {0} time: {1} s'.format(
                    epoch + 1,
                    datetime.now() - epoch_time))
                print('\t\ttrain loss: {0:.3f}'.format(train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop or self.args.dry_run:
                print('early stopping!')
                break
            self.writer.close()

    def validate(self, loader, criterion):
        val_loss = 0
        self.model.eval()
        val_steps = len(loader)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                batch_x.double().to(self.device)
                batch_y.double().to(self.device)
                pred = self.model(batch_x)
                pred = pred[:,
                            -self.args.pred_len:, :].double().to(self.device)
                rmse_loss = torch.sqrt(criterion(pred, batch_y))
                label = 'val' if loader.dataset.flag == 'val' else 'test'
                self.writer.add_scalar(f'loss {label}/iter', rmse_loss, i)
                val_loss += rmse_loss.item()
        val_loss /= val_steps
        self.model.train()
        return val_loss

    def test(self):
        pass

    def predict(self):
        pass
