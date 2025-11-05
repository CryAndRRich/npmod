import torch
import torch.nn as nn
from ..rnn import RecurNet

class LSTMNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.1) -> None:
        """
        LSTM - Long Short-Term Memory Network

        Parameters:
            input_size: Number of input features
            hidden_size: Number of hidden units in each LSTM layer
            output_size: Dimension of the output layer
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of LSTM layers
            bidirectional: Whether to use a bidirectional LSTM
            dropout: Dropout rate between LSTM layers
        """
        super().__init__()

        self.lstm = nn.LSTM(
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
        out, _ = self.lstm(x)
        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out


class LSTM(RecurNet):
    def init_network(self) -> None:
        self.network = LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Long Short-Term Memory (LSTM)"
