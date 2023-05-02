import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch


class StandardScaler():

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class EarlyStopping:

    def __init__(self, patience: int, verbose: bool = True, delta: int = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path='./runs'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.path = os.path.join(path)
            self.save_checkpoint(val_loss, model, self.path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f})',
                'saving model',
                sep='\n')
        os.makedirs(path, exist_ok=True)
        torch.save(model, os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def plot_prediction(history,
                    true,
                    preds,
                    pred_range,
                    name: str,
                    save: bool = True,
                    show: bool = True):
    plt.figure()
    plt.plot(history, label='history')
    plt.plot(pred_range, true, label='ground truth')
    plt.plot(pred_range, preds, label='prediction')
    plt.legend()
    if save: plt.savefig(name, bbox_inches='tight')
    if show: plt.show()
    plt.close()
