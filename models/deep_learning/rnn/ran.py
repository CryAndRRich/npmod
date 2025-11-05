import torch
import torch.nn as nn
from ..rnn import RecurNet

class RANCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the RAN cell

        Parameters:
            input_size: Size of input vector at each time step
            hidden_size: Number of hidden units
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Forget gate parameters
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

        # Input gate parameters
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

        # Input projection
        self.W_x = nn.Linear(input_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step of the RAN cell

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Previous hidden state

        Returns:
            h_t: Current hidden state
        """
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        x_proj = self.W_x(x_t)
        h_t = f_t * h_prev + i_t * x_proj
        return h_t


class RANNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the RAN network

        Parameters:
            input_size: Dimension of the input features
            hidden_size: Number of hidden units in the RAN cell
            output_size: Output dimension
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of RAN layers
            dropout: Dropout rate applied after recurrent processing
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Stacked RAN layers
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(RANCell(in_size, hidden_size))
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


class RAN(RecurNet):
    def init_network(self) -> None:
        self.network = RANNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Recurrent Additive Network (RAN)"
