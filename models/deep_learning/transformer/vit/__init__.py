from typing import Optional

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 image_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 3, 
                 embed_dim: int = 768) -> None:
        """
        Patch Embedding using Conv2d to extract patches and project to embedding dimension
        
        Parameters:
            image_size: Size of the input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_chans: Number of input channels
            embed_dim: Dimension of the embedding space
        """
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        # Use conv to project patches
        self.proj = nn.Conv2d(in_channels=in_chans, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) 
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 heads: int = 8, 
                 dropout: float = 0.0) -> None:
        """
        Multi-Head Self-Attention module
        
        Parameters:
            dim: Dimension of the input features
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Self-Attention
        
        Parameters:
            x: Input tensor 
            attn_mask: Optional attention mask 
        """
        B, N, D = x.shape
        qkv = self.qkv(x) 
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  
        out = out.transpose(1, 2).reshape(B, N, D) 
        out = self.out(out)
        out = self.proj_drop(out)
        return out
    

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

from .ViT import ViT
from .DeiT import DeiT
from .Swin import Swin