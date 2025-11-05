import torch
import torch.nn as nn
from ..rnn import RecurNet

class MGUCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the MGU cell

        Parameters:
            input_size: Dimension of the input vector
            hidden_size: Number of hidden units in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute one time step of the MGU cell

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Hidden state from previous time step

        Returns:
            h_t: Updated hidden state
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        z_t = torch.sigmoid(self.W_z(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class MGUNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the MGU network

        Parameters:
            input_size: Dimension of the input features
            hidden_size: Number of hidden units in the MGU cell
            output_size: Output dimension
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of MGU layers
            dropout: Dropout rate applied after recurrent processing
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(MGUCell(in_size, hidden_size))
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


class MGU(RecurNet):
    def init_network(self) -> None:
        self.network = MGUNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Minimal Gated Unit (MGU)"
