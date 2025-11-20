import torch
import torch.nn as nn

class LayerNorm(nn.LayerNorm):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5) -> None:
        """
        Layer Normalization
        """
        super().__init__(d_model, eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-8) -> None:
        """
        Root Mean Square Layer Normalization
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + self.eps)


class ScaleNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5) -> None:
        """
        Scale Normalization
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(d_model ** 0.5))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


class AdaNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 beta: float = 1.0, 
                 eps: float = 1e-5) -> None:
        """
        Adaptive Normalization
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = beta
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        x_hat = (x - mean) / std
        norm = (1 + self.beta * torch.tanh(x_hat))
        return self.scale * (self.alpha * norm) + self.bias