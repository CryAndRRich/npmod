import torch
import torch.nn as nn
from ..rnn import RecurNet

class JANETCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the JANET cell

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate and candidate cell state
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t
            h_prev: Previous hidden state
            c_prev: Previous cell state

        Returns:
            h_t: Current hidden state
            c_t: Current cell state
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        f_t = torch.sigmoid(self.W_f(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        c_t = f_t * c_prev + (1 - f_t) * c_tilde
        h_t = torch.tanh(c_t)
        return h_t, c_t


class JANETNet(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the JANET network

        Parameters:
            input_size: Input feature dimension
            hidden_size: Number of hidden units
            output_size: Output dimension
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of JANET layers
            dropout: Dropout rate applied after recurrent processing
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(JANETCell(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            for l in range(self.num_layers):
                h[l], c[l] = self.layers[l](x_t, h[l], c[l])
                x_t = h[l]
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  
        outputs = self.dropout(outputs[:, -self.forecast_horizon:, :])
        out = self.fc(outputs) 
        return out


class JANET(RecurNet):
    def init_network(self) -> None:
        self.network = JANETNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Just Another NETwork (JANET)"
