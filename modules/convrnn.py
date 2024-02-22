from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ConvLSTM", "ConvGRU", "ConvRNN"]


class ConvRNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']
    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: nn.Conv2d
    weight_hh: nn.Conv2d

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, kernel_size=(1, 1),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Conv2d(input_size, hidden_size * num_chunks, kernel_size=kernel_size,
                                   padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=bias)
        self.weight_hh = nn.Conv2d(hidden_size, hidden_size * num_chunks, kernel_size=kernel_size,
                                   padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=bias)
        self.num_chunks = num_chunks
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class ConvRNNCell(ConvRNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, kernel_size: Tuple = (1, 1),
                 nonlinearity: str = "tanh", device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvRNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1, kernel_size=kernel_size,
                                          **factory_kwargs)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        i = self.weight_ih(input)
        hidden = self.weight_hh(hx) + i
        if self.nonlinearity == "tanh":
            ret = torch.tanh(hidden)
        elif self.nonlinearity == "relu":
            ret = torch.relu(hidden)
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, kernel_size: Tuple = (1, 1),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4,
                                           kernel_size=kernel_size, **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        # c' and h'
        h, c = hx
        gates = self.weight_ih(input) + self.weight_hh(h)
        i, f, g, o = torch.split(gates, self.hidden_size, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_ = f * c + i * g
        h_ = o * torch.tanh(c_)
        return (h_, c_)


class ConvGRUCell(ConvRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, kernel_size: Tuple = (1, 1),
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, kernel_size=kernel_size,
                                          **factory_kwargs)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        r1, z1, n1 = torch.split(self.weight_ih(input), self.hidden_size, dim=1)
        r2, z2, n2 = torch.split(self.weight_hh(hx), self.hidden_size, dim=1)
        r = torch.sigmoid(r1 + r2)
        z = torch.sigmoid(z1 + z2)
        n = torch.tanh(n1 + n2 * r)
        h_ = (1 - z) * n + z * hx
        return h_


class ConvRNNBase(nn.Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'kernel_size']
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    kernel_size: int

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0., bidirectional: bool = False, kernel_size: Union[Tuple, int] = (1, 1),
                 device=None, dtype=None, circular=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ConvRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.kernel_size = kernel_size
        self.circular = circular
        num_directions = 2 if bidirectional else 1
        self.modules_forward = nn.ModuleList()
        self.modules_reverse = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                if self.mode == "LSTM":
                    cell = ConvLSTMCell(layer_input_size, hidden_size, bias, kernel_size, device, dtype)
                elif self.mode == "GRU":
                    cell = ConvGRUCell(layer_input_size, hidden_size, bias, kernel_size, device, dtype)
                elif self.mode == "RNN_TANH":
                    cell = ConvRNNCell(layer_input_size, hidden_size, bias, kernel_size, nonlinearity="tanh",
                                       device=None, dtype=None)
                elif self.mode == "RNN_RELU":
                    cell = ConvRNNCell(layer_input_size, hidden_size, bias, kernel_size, nonlinearity="relu",
                                       device=None, dtype=None)
                else:
                    raise ValueError("Unrecognized RNN mode: " + mode)
                if direction == 0:
                    self.modules_forward.append(cell)
                else:
                    self.modules_reverse.append(cell)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 5
        b, t, c, h, w = x.shape
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        xs = list(torch.unbind(x, dim=1))
        num_directions = 2 if self.bidirectional else 1
        if self.mode == "LSTM":
            h_zeros = torch.zeros(self.num_layers * num_directions, b, self.hidden_size, h, w, dtype=x.dtype,
                                  device=x.device)
            c_zeros = torch.zeros(self.num_layers * num_directions, b, self.hidden_size, h, w, dtype=x.dtype,
                                  device=x.device)
            hx = torch.stack([h_zeros, c_zeros], dim=1)
        else:
            hx = torch.zeros(self.num_layers * num_directions, b, self.hidden_size, h, w, dtype=x.dtype,
                             device=x.device)
        if self.circular:
            xs = [xs[-1], *xs, xs[0]]
        layer_input = xs
        layer_output: List[Union[Tuple, Tensor]] = [(0,)] * len(xs)  # 每一层的正向的hidden（cell）的输出
        for layer in range(self.num_layers):
            hidden_pos = layer * 2 if self.bidirectional else layer
            for time_step in range(len(xs)):
                if time_step == 0:
                    out = self.modules_forward[layer](layer_input[time_step], hx[hidden_pos])
                else:
                    out = self.modules_forward[layer](layer_input[time_step], layer_output[time_step - 1])
                layer_output[time_step] = out
            if self.circular:
                t = layer_output[0]
                layer_output[0] = layer_output[-1]
                layer_output[-1] = t
            if self.bidirectional:
                hidden_pos += 1
                layer_reverse_output = [0] * len(xs)
                for time_step in range(len(xs) - 1, -1, -1):
                    if time_step == len(xs) - 1:
                        out = self.modules_reverse[layer](layer_input[time_step], hx[hidden_pos])
                    else:
                        out = self.modules_reverse[layer](layer_input[time_step], layer_output[time_step + 1])
                    layer_reverse_output[time_step] = out
                if self.circular:
                    t = layer_reverse_output[0]
                    layer_reverse_output[0] = layer_reverse_output[-1]
                    layer_reverse_output[-1] = t
                for i, (forward_out, backward_out) in enumerate(zip(layer_output, layer_reverse_output)):
                    if self.mode == "LSTM":
                        layer_input[i] = torch.cat([forward_out[0], backward_out[0]], dim=1)

                    else:
                        layer_input[i] = torch.cat([forward_out, backward_out], dim=1)
            else:
                if self.mode == "LSTM":
                    layer_input = [v[0] for v in layer_output]
                else:
                    layer_input = layer_output
        res = torch.stack(layer_input, dim=1)
        if self.circular:
            res = res[:, 1:-1]
        return res


class ConvRNN(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(ConvRNN, self).__init__(mode, *args, **kwargs)


class ConvLSTM(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvLSTM, self).__init__('LSTM', *args, **kwargs)


class ConvGRU(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvGRU, self).__init__('GRU', *args, **kwargs)
