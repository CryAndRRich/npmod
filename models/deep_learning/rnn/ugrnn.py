import torch
import torch.nn as nn
from ..rnn import RecurNet

class UGRNNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int) -> None:
        """
        Initialize the UGRNN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Size of the hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single time step

        Parameters:
            x_t: Input tensor at time step t 
            h_prev: Hidden state from the previous time step 

        Returns:
            h_t: Updated hidden state 
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        z_t = torch.sigmoid(self.W_z(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        h_t = z_t * h_prev + (1 - z_t) * h_tilde
        return h_t


class UGRNNLayer(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.1) -> None:
        """
        UGRNN Layer consisting of UGRNN cells

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units
            dropout: Dropout rate
        """
        super().__init__()
        self.cell = UGRNNCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                h_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        _, seq_len, _ = x.shape
        h_t = h_0
        outputs = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out), h_t
    

class UGRNNNet(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the UGRNN network
        
        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in UGRNN cell
            output_size: Number of output classes
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of UGRNN layers
            dropout: Dropout rate between UGRNN layers
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(UGRNNLayer(in_size, hidden_size, dropout))
        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass over an input sequence

        Parameters:
            x: Input tensor 

        Returns:
            out: Logits for each class 
        """
        batch_size = x.size(0)
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        out = x
        for i, layer in enumerate(self.layers):
            out, h[i] = layer(out, h[i])

        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out
    

class UGRNN(RecurNet):
    def init_network(self) -> None:
        self.network = UGRNNNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Update Gate RNN (UGRNN)"
