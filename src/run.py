import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import TestRigData
from models import LSTMModel
from tools import EarlyStopping, plot_prediction


class Run:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)

    def _initialize_criterion(self):
        self.criterion = nn.MSELoss()

    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.learning_rate)

    def _initialize_scheduler(self, optimizer):
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.args.gamma)

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

    def _initialize_model(self):
        model = LSTMModel
        self.model = model(input_size=self.train_data.x.shape[-1],
                           output_size=self.train_data.y.shape[-1],
                           hidden_size=self.args.hidden_size,
                           num_layers=self.args.num_layers,
                           dropout=self.args.dropout,
                           batch_size=self.args.batch_size)

    def _log_epoch_start(self, epoch):
        logging.info(f'epoch: {epoch + 1}')
        self.epoch_path = os.path.join(self.path, 'epochs', f'{epoch+1}')
        os.makedirs(self.epoch_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.epoch_path)

    def _log_epoch_end(self, epoch, epoch_start_time, train_loss, val_loss,
                       test_loss):
        self.writer.add_scalars(
            'loss/epoch',
            {
                'train': train_loss,
                'val': val_loss,
                'test': test_loss
            },
            epoch + 1,
        )
        self.writer.add_scalar('lr/epoch',
                               float(self.scheduler.get_last_lr()[0]),
                               epoch + 1)
        with open(os.path.join(self.epoch_path, 'metrics.txt'), 'w') as fp:
            fp.write(f'train {train_loss}\nval {val_loss}\ntest {test_loss}')
        logging.info(
            f'epoch {epoch + 1} time: {datetime.now() - epoch_start_time} s')
        logging.info(
            f'train loss: {train_loss:.3f} | val loss: {val_loss:.3f} | test loss: {test_loss:.3f}'
        )

    def _log_iteration(self, train_steps, epoch, i, batch_x, batch_y, pred,
                       rmse_loss):
        logging.info('\titer: {0:>5d}/{1:>5d} | loss: {2:.3f}'.format(
            i, train_steps, rmse_loss.item()))
        pred_range = torch.arange(self.args.seq_len,
                                  self.args.seq_len + self.args.pred_len)
        history = self.train_data.scaler.inverse_transform(
            torch.squeeze(batch_x[0, -self.args.seq_len:, :]))
        trues = torch.squeeze(batch_y[0, -self.args.pred_len:, :]).detach()
        preds = torch.squeeze(pred[0, :, :]).detach()
        fig_path = os.path.join(self.path, 'epochs', f'{epoch+1}')
        fig_name = f'{i}.png'
        fig = plot_prediction(history=history,
                              true=trues,
                              preds=preds,
                              pred_range=pred_range,
                              path=fig_path,
                              name=fig_name,
                              to_tb=True,
                              show=False,
                              save=False)
        if fig:
            self.writer.add_figure(f'{epoch+1}_{fig_name}', fig, i)

    def _log_args_model(self):
        logging.info('arguments:')
        with open(os.path.join(self.path, 'args.txt'), 'w') as fp:
            for k, v in vars(self.args).items():
                fp.write(f'{k} {v}\n')
                logging.info(f'\t{k:<15}{v:<15}')
        logging.info('model:')
        logging.info(self.model)

    def _initilize_logging(self):
        day_stamp = datetime.now().strftime('%Y%m%d')
        time_stamp = datetime.now().strftime('%H%M%S')
        self.path = os.path.join(os.environ['RUNS_PATH'], day_stamp,
                                 time_stamp)
        os.makedirs(self.path, exist_ok=True)
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%H:%M:%S',
                            handlers=[
                                logging.FileHandler(
                                    os.path.join(self.path, 'log.txt')),
                                logging.StreamHandler()
                            ],
                            level=logging.INFO)

    def _initialize_data(self):
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.val_data, self.val_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')

    def _finalize_writer(self):
        self.writer.flush()
        self.writer.close()

    def train(self):
        self._initialize_data()
        self._initialize_model()
        self._initialize_criterion()
        self._initialize_optimizer()
        self._initialize_scheduler(self.optimizer)
        self._initilize_logging()
        self._log_args_model()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True)
        for epoch in range(self.args.epochs):
            self._log_epoch_start(epoch)
            train_losses = []
            epoch_start_time = datetime.now()
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.float(), batch_y.float()
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                rmse_loss = torch.sqrt(self.criterion(pred, batch_y))
                self.writer.add_scalar('loss/train/iter', rmse_loss, i)
                train_losses.append(rmse_loss.item())
                if i % self.args.log_interval == 0:
                    self._log_iteration(train_steps, epoch, i, batch_x,
                                        batch_y, pred, rmse_loss)
                rmse_loss.backward()
                self.optimizer.step()
                if self.args.dry_run: break
            train_loss = np.mean(train_losses)
            val_loss = self.evaluate(self.val_loader, self.criterion)
            test_loss = self.evaluate(self.test_loader, self.criterion)
            self._log_epoch_end(epoch, epoch_start_time, train_loss, val_loss,
                                test_loss)
            early_stopping(val_loss, epoch + 1, self.model, self.path)
            self.scheduler.step()
            if early_stopping.early_stop or self.args.dry_run:
                logging.info('early stopping!')
                break
        self._finalize_writer()
        self._load_best_model()
        return self.model

    def _load_best_model(self):
        best_model_path = os.path.join(self.path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

    def evaluate(self, loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                rmse_loss = torch.sqrt(criterion(pred, batch_y))
                self.writer.add_scalar(f'loss/{loader.dataset.flag}/iter',
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
                    inputs = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]),
                                        axis=0)
                    pred = np.concatenate((inputs[0, :, -1], pred[0, :, -1]),
                                          axis=0)
                    plot_prediction(gt, pred,
                                    os.path.join(test_folder, f'{i}.pdf'))

    def predict(self):
        pass
