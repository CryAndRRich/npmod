import torch
import torch.nn as nn
from ..rnn import RecurNet

class IndRNNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize an IndRNN cell

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_weights = nn.Linear(input_size, hidden_size)
        self.recurrent_weights = nn.Parameter(torch.ones(hidden_size))  # Element-wise recurrent weights
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass for one time step

        Parameters:
            x_t: Input tensor at time step t
            h_prev: Previous hidden state tensor

        Returns:
            h_t: Current hidden state tensor
        """
        input_term = self.input_weights(x_t)
        recurrent_term = self.recurrent_weights * h_prev
        h_t = torch.relu(input_term + recurrent_term + self.bias)
        return h_t


class IndRNNNet(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Parameters:
            input_size: Dimension of the input features
            hidden_size: Number of hidden units in the IndRNN cell
            output_size: Output dimension
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of IndRNN layers
            dropout: Dropout rate applied after recurrent processing
        """
        super().__init__()
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(IndRNNCell(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                h[l] = self.layers[l](x_t, h[l])
                x_t = h[l] 
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        outputs = self.dropout(outputs[:, -self.forecast_horizon:, :])
        out = self.fc(outputs)
        return out


class IndRNN(RecurNet):
    def init_network(self) -> None:
        self.network = IndRNNNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Independently Recurrent Neural Network (IndRNN)"
