import os
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import TestRigData
from models import LSTM
from tools import EarlyStopping, visual


class Experiment:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)

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
                     batch_size=self.args.batch_size)

    def train(self, setting):
        self.train_data, train_loader = self._get_data(flag='train')
        self.val_data, val_loader = self._get_data(flag='val')
        self.test_data, test_loader = self._get_data(flag='test')
        self.model = self._build_model()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
        self.path = os.path.join('./runs', setting)
        os.makedirs(self.path, exist_ok=True)
        print('arguments:')
        with open(os.path.join(self.path, 'params.txt'), 'a') as fp:
            for k, v in vars(self.args).items():
                if all((k, v)):
                    print('\t{}: {}'.format(k, v))
                    fp.write('{}: {}\n'.format(k, v))
        for epoch in range(self.args.epochs):
            print('epoch: {}'.format(epoch + 1))
            train_losses = []
            self.model.train()
            epoch_time = datetime.now()
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.path, f'{epoch+1}'))
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                model_optim.zero_grad()
                pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                rmse_loss = torch.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar('loss train/iter', rmse_loss, i)
                train_losses.append(rmse_loss.item())
                if i % self.args.log_interval == 0:
                    print('\titer: {0:>5d}/{1:>5d} | loss: {2:.3f}'.format(
                        i, train_steps, rmse_loss.item()))
                rmse_loss.backward()
                model_optim.step()
                if self.args.dry_run: break
            train_loss = np.mean(train_losses)
            val_loss = self.evaluate(val_loader, criterion)
            test_loss = self.evaluate(test_loader, criterion)
            print('epoch {0} time: {1} s'.format(epoch + 1,
                                                 datetime.now() - epoch_time))
            print(
                'train loss: {0:.3f} | val loss: {1:.3f} | test loss: {2:.3f}'.
                format(train_loss, val_loss, test_loss))
            early_stopping(val_loss, self.model, self.path)
            self.writer.flush()
            if early_stopping.early_stop or self.args.dry_run:
                print('early stopping!')
                break
            self.writer.close()
        best_model_path = os.path.join(self.path, 'checkpoint.pth')
        self.model = torch.load(best_model_path)
        return self.model

    def evaluate(self, loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                rmse_loss = np.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar(f'loss {loader.dataset.flag}/iter',
                                       rmse_loss, i)
                losses.append(rmse_loss.item())
        loss = np.mean(losses)
        self.model.train()
        return loss

    def test(self):
        self.test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        test_folder = os.path.join(self.path, 'test_results')
        os.makedirs(test_folder, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                if i % self.args.log_interval == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]),
                                        axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]),
                                        axis=0)
                    visual(gt, pd, os.path.join(test_folder, f'{i}.pdf'))

    def predict(self):
        pass
