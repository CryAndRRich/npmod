import torch
import torch.nn as nn
from ..rnn import RecurNet

class RHNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 depth: int = 2) -> None:
        """
        Initialize the RHN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Number of hidden units
            depth: Number of highway layers in the cell
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth

        self.input_transform = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        ])
        self.transform_gates = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        ])

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step of the RHN cell

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Previous hidden state

        Returns:
            h: Current hidden state
        """
        h = self.input_transform(x_t)

        for i in range(self.depth):
            t_gate = torch.sigmoid(self.transform_gates[i](h))
            h_layer = torch.tanh(self.layers[i](h))
            h = t_gate * h_layer + (1 - t_gate) * h

        return h + h_prev # Residual connection


class RHNLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 depth: int = 2,
                 dropout: float = 0.1) -> None:
        """
        One RHN recurrent layer
        
        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in RHN cell
            depth: Number of highway layers in each RHN cell
            dropout: Dropout rate between RHN layers
        """
        super().__init__()
        self.cell = RHNCell(input_size, hidden_size, depth)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                h_0: torch.Tensor) -> torch.Tensor:
        outputs = []
        h_t = h_0
        for t in range(x.size(1)):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out), h_t
    

class RHNNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 depth: int = 2,
                 dropout: float = 0.1) -> None:
        """
        Initialize the RHN network

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in RHN cell
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of RHN layers
            depth: Number of highway layers in each RHN cell
            dropout: Dropout rate between RHN layers
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(RHNLayer(in_size, hidden_size, depth, dropout))
        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RHN network

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

class RHN(RecurNet):
    def init_network(self) -> None:
        self.network = RHNNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            depth=getattr(self, "depth", 2),
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Recurrent Highway Network (RHN)"
