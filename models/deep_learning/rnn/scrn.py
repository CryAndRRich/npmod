import torch
import torch.nn as nn
from ..rnn import RecurNet

class SCRNCell(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 context_size: int, 
                 alpha: float = 0.95) -> None:
        """
        Initialize the SCRN cell

        Parameters:
            input_size: Size of the input vector at each time step
            hidden_size: Size of the hidden state
            context_size: Size of the context state
            alpha: Smoothing factor for context update (0 < alpha < 1)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.alpha = alpha

        self.W_x = nn.Linear(input_size, context_size)
        self.U_c = nn.Linear(context_size, hidden_size, bias=False)
        self.V_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, 
                x_t: torch.Tensor, 
                h_prev: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step of SCRN

        Parameters:
            x_t: Input tensor at current time step
            h_prev: Previous hidden state
            c_prev: Previous context state

        Returns:
            h_t: Current hidden state
            c_t: Current context state
        """
        c_t = (1 - self.alpha) * c_prev + self.alpha * self.W_x(x_t)
        h_t = torch.tanh(self.U_c(c_t) + self.V_h(h_prev))
        return h_t, c_t


class SCRNLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 context_size: int,
                 alpha: float = 0.95,
                 dropout: float = 0.1) -> None:
        """
        SCRN Layer consisting of SCRN cells

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in the SCRN cell
            context_size: Size of the context vector
            alpha: Smoothing factor for context update
            dropout: Dropout rate after the SCRN layer
        """
        super().__init__()
        self.cell = SCRNCell(input_size, hidden_size, context_size, alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                h_0: torch.Tensor, 
                c_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        outputs = []
        h_t, c_t = h_0, c_0
        for t in range(x.size(1)):
            h_t, c_t = self.cell(x[:, t, :], h_t, c_t)
            outputs.append(h_t.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return self.dropout(out), h_t, c_t


class SCRNNet(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 context_size: int = 32,
                 alpha: float = 0.95,
                 forecast_horizon: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.1) -> None:
        """
        Initialize the SCRN network

        Parameters:
            input_size: Size of the input features
            hidden_size: Number of hidden units in SCRN cell
            output_size: Number of output classes
            context_size: Size of the context vector
            alpha: Smoothing factor for context update
            forecast_horizon: Number of time steps to forecast
            num_layers: Number of SCRN layers
            dropout: Dropout rate between SCRN layers
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(SCRNLayer(in_size, hidden_size, context_size, alpha, dropout))
        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SCRN network

        Parameters:
            x: Input tensor 

        Returns:
            out: Output tensor 
        """
        batch_size = x.size(0)

        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.context_size) for _ in range(self.num_layers)]

        out = x
        for i, layer in enumerate(self.layers):
            out, h[i], c[i] = layer(out, h[i], c[i])

        out = self.fc(out[:, -self.forecast_horizon:, :])
        return out


class SCRN(RecurNet):
    def init_network(self) -> None:
        self.network = SCRNNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            context_size=getattr(self, "context_size", 32),
            alpha=getattr(self, "alpha", 0.95),
            forecast_horizon=self.forecast_horizon,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def __str__(self) -> str:
        return "Structurally Constrained Recurrent Network (SCRN)"
