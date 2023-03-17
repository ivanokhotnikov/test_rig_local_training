import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 batch_size: int, output_size: int, **kwargs):
        super(LSTM, self).__init__()
        # rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2) # input_size - number of expected features in the input x, hidden_size -  number of features in the hidden state h

        # Inputs:
        # input = torch.randn(5, 3, 10) # [batch_size, input_size] for unbatched input, [seq_len, batch_size, input_size] when batch_first=False, [batch_size, seq_len, input_size] when batch_first=True

        # h0 = torch.randn(2, 3, 20) # [D * num_layers, hidden_size] for unbatched input, [D * num_layers, barch_size, hidden_size] , D = 2 if birectional=True, otherwise 1
        # c0 = torch.randn(2, 3, 20)
        # output, (hn, cn) = rnn(input, (h0, c0))
        self.input_size = input_size  # number of expected features in the input
        self.hidden_size = hidden_size  # number of features in the hidden state
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dtype=torch.float)
        self.linear = nn.Linear(self.hidden_size,
                                self.output_size,
                                dtype=torch.float)
        self.h0 = torch.randn(self.num_layers,
                              self.batch_size,
                              self.hidden_size,
                              dtype=torch.float)
        self.c0 = torch.randn(self.num_layers,
                              self.batch_size,
                              self.hidden_size,
                              dtype=torch.float)

    def forward(self, input):
        # input [self.batch_size, self.seq_len, self.input_size]
        output, (hn, cn) = self.lstm(input, (self.h0, self.c0))
        output = self.linear(output)
        return output
