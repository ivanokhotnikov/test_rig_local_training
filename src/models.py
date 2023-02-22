import torch
from torch import nn
from torch.nn.functional import F


class LSTM(nn.Module):

    def __init__(self, configs):
        super(LSTM, self).__init__()
        # lstm = nn.LSTM(1, 20, 2, batch_first=True)
        # h0 = torch.randn(2, 64, 20)
        # c0 = torch.randn(2, 64, 20)
        # output, (hn, cn) = lstm(x.float(), (h0, c0))
