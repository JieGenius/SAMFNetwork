import torch
import torch.nn as nn
from modules.convrnn import ConvRNN, ConvGRU, ConvLSTM


class SAL(nn.Module):
    def __init__(self, mode, sector_count=8, input_dim=256, hidden_dim=128, num_layers=1, seq_pad=False,
                 bidirectional=True, kernel_size=(3, 3)):
        super(SAL, self).__init__()
        self.sector_count = sector_count
        self.seq_pad = seq_pad
        self.input_size = input_dim
        self.mode = mode
        params = dict(input_size=input_dim,
                      hidden_size=hidden_dim,
                      bias=True,
                      batch_first=True,
                      num_layers=num_layers,
                      bidirectional=bidirectional,
                      kernel_size=kernel_size,
                      circular=seq_pad)
        if mode == "lstm":
            self.rnn = ConvLSTM(**params)
        elif mode == "gru":
            self.rnn = ConvGRU(**params)
        elif mode == "rnn":
            self.rnn = ConvRNN(**params)
        else:
            raise ValueError()

    def forward(self, x: torch.Tensor):
        if self.mode != "transformer":
            b, c, h, w = x.shape
            x = x.split(h // self.sector_count, dim=2)
            x = torch.stack(x, dim=1)
            x = self.rnn(x)
            b, sc, c, h, w = x.shape  # sc: sector count
            x = x.permute(0, 2, 1, 3, 4).reshape(b, c, -1, w)
        else:
            x = self.rnn(x)
        return x
