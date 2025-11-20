import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN_ReLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int) -> None:
        """
        Feed-Forward Network with ReLU activation
        
        Parameters:
            d_model: Dimension of the model
            dim_ff: Dimension of the feed-forward layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFN_GELU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int) -> None:
        """
        Feed-Forward Network with GELU activation
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFN_GEGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int) -> None:
        """
        Feed-Forward Network with GELU Gated Linear Unit
        """
        super().__init__()
        self.proj = nn.Linear(d_model, dim_ff * 2)
        self.out = nn.Linear(dim_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x_val, gate = x_proj.chunk(2, dim=-1)
        return self.out(F.gelu(gate) * x_val)


class FFN_SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int) -> None:
        """
        Feed-Forward Network with SwiGLU activation
        """
        super().__init__()
        self.proj = nn.Linear(d_model, dim_ff * 2)
        self.out = nn.Linear(dim_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x_val, gate = x_proj.chunk(2, dim=-1)
        return self.out(F.silu(gate) * x_val)


class FFN_GLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int) -> None:
        """
        Feed-Forward Network with standard GLU activation
        """
        super().__init__()
        self.proj = nn.Linear(d_model, dim_ff * 2)
        self.out = nn.Linear(dim_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x_val, gate = x_proj.chunk(2, dim=-1)
        return self.out(torch.sigmoid(gate) * x_val)


class FFN_Conformer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int, 
                 dropout: float = 0.1) -> None:
        """
        Conformer-style Feed-Forward Network
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(dim_ff, dim_ff, kernel_size=3, padding=1, groups=dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.swish(self.linear1(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.swish(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + 0.5 * residual


class FFN_DropConnect(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dim_ff: int, 
                 dropconnect: float = 0.2) -> None:
        """
        Feed-Forward Network with DropConnect regularization
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)
        self.dropconnect = dropconnect

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = F.dropout(self.fc1.weight, p=self.dropconnect, training=self.training)
        w2 = F.dropout(self.fc2.weight, p=self.dropconnect, training=self.training)
        hidden = F.relu(F.linear(x, w1, self.fc1.bias))
        return F.linear(hidden, w2, self.fc2.bias)
