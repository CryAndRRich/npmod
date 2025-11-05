import torch
import torch.nn as nn
from ..rnn import RecurNet

class YamRNNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        YamRNN cell for sequence modeling

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = nn.Linear(input_size, hidden_size)
        self.w_hh1 = nn.Linear(hidden_size, hidden_size)
        self.w_hh2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t
            h_prev: Hidden state tensor from the previous time step

        Returns:
            h_t: Updated hidden state tensor
        """
        v1 = torch.tanh(self.w_ih(x_t) + self.w_hh1(h_prev))
        v2 = torch.tanh(self.w_ih(x_t) + self.w_hh2(h_prev))
        h_t = (1 - h_prev) * v1 + h_prev * v2
        return h_t


class YamRNNLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.1) -> None:
        """
        YamRNN Layer consisting of YamRNN cells

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units
            dropout: Dropout rate
        """
        super().__init__()
        self.cell = YamRNNCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                h_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        h_t = h_0
        for t in range(x.size(1)):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out), h_t
    

class YamRNNNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the YamRNN network

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in the YamRNN cell
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of YamRNN layers
            dropout: Dropout rate between YamRNN layers
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(YamRNNLayer(in_size, hidden_size, dropout))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass over the sequence

        Parameters:
            x: Input tensor

        Returns:
            out: Output tensor
        """
        batch_size = x.size(0)
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        out = x
        for i, layer in enumerate(self.layers):
            out, h[i] = layer(out, h[i])

        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out

class YamRNN(RecurNet):
    def init_network(self) -> None:
        self.network = YamRNNNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "YamRNN"
