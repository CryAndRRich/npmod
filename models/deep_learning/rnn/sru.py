import torch
import torch.nn as nn
from ..rnn import RecurNet

class SRUCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, 3 * hidden_size)

        # Input projection
        self.W_x = nn.Linear(input_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SRU cell

        Parameters:
            x_t: Current input at time t 
            c_prev: Previous memory cell 

        Returns:
            h_t: Current hidden state 
            c_t: Current memory cell 
        """
        x_proj = self.linear(x_t)
        x_tilde, f_t, r_t = torch.chunk(x_proj, 3, dim=-1)

        f_t = torch.sigmoid(f_t)
        r_t = torch.sigmoid(r_t)

        c_t = f_t * c_prev + (1 - f_t) * x_tilde
        
        x_res = self.W_x(x_t)
        h_t = r_t * torch.tanh(c_t) + (1 - r_t) * x_res
        return h_t, c_t


class SRULayer(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.1) -> None:
        """
        SRU Layer consisting of SRU cells

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units
            dropout: Dropout rate
        """
        super().__init__()
        self.cell = SRUCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor,
                c_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        outputs = []
        c_t = c_0
        for t in range(x.size(1)):
            h_t, c_t = self.cell(x[:, t, :], c_t)
            outputs.append(h_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out), c_t


class SRUNet(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in the SRU layer
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of SRU layers
            dropout: Dropout rate between SRU layers
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(SRULayer(in_size, hidden_size, dropout))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SRU network

        Parameters:
            x: Input tensor 
        
        Returns:
            out: Output tensor
        """
        batch_size = x.size(0)

        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        out = x

        for i, layer in enumerate(self.layers):
            out, c[i] = layer(out, c[i])

        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out

class SRU(RecurNet):
    def init_network(self) -> None:
        self.network = SRUNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Simple Recurrent Unit (SRU)"
