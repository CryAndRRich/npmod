import torch
import torch.nn as nn
from ..rnn import RecurNet

class GRUNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.1) -> None:
        """
        GRU - Gated Recurrent Unit Network
        
        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in the GRU layer
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of GRU layers
            bidirectional: If True, use a bidirectional GRU
            dropout: Dropout rate between GRU layers
        """
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out

class GRU(RecurNet):
    def init_network(self) -> None:
        self.network = GRUNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Gated Recurrent Unit (GRU)"
