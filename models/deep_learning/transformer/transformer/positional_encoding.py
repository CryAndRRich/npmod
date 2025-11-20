import math
from typing import List
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 5000) -> None:
        """
        Implements the sinusoidal positional encoding

        Parameters:
            d_model: Dimension of the model
            max_len: Maximum length of the input sequences
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 5000) -> None:
        """
        Implements learned positional encoding
        """
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = min(x.size(1), self.pe.num_embeddings)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x[:, :seq_len, :] + self.pe(pos)


class RelativePositionalBias(nn.Module):
    def __init__(self, 
                 num_heads: int, 
                 max_distance: int = 128) -> None:
        """
        Implements relative positional bias
        
        Parameters:
            num_heads: Number of attention heads
            max_distance: Maximum relative distance to consider
        """
        super().__init__()
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_len, k_len = x.size(1), x.size(1)
        context_position = torch.arange(q_len)[:, None]
        memory_position = torch.arange(k_len)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.clamp(-self.max_distance, self.max_distance)
        relative_position = relative_position + self.max_distance
        values = self.relative_bias(relative_position)
        return values.permute(2, 0, 1).to(dtype=torch.float32)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Implements Rotary Positional Embedding
        
        Parameters:
            dim: Dimension of the model
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
        cos = cos[:, :seq_len, None, :]
        sin = sin[:, :seq_len, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + x_rot * sin


class ALiBiBias(nn.Module):
    def __init__(self, num_heads: int) -> None:
        """
        Implements ALiBi positional bias
        
        Parameters:
            num_heads: Number of attention heads
        """
        super().__init__()
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n: int) -> torch.Tensor:
        """
        Get slopes for ALiBi bias
        """
        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return torch.tensor(get_slopes_power_of_2(closest_power_of_2) +
                                 get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_len, k_len = x.size(1), x.size(1)
        bias = torch.arange(k_len).unsqueeze(0) - torch.arange(q_len).unsqueeze(1)
        bias = bias.unsqueeze(0) * -self.slopes.unsqueeze(1).unsqueeze(1)
        return bias.to(dtype=torch.float32)
