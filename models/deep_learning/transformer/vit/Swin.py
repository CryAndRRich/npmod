from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vit import *

class PatchMerging(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int) -> None:
        """
        Patch Merging Layer for downsampling
        
        Parameters:
            input_dim: Dimension of the input features
            output_dim: Dimension of the output features
        """
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, 
                x: torch.Tensor, 
                H: int, 
                W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass for Patch Merging
        
        Parameters:
            x: Input tensor
            H: Height of the feature map
            W: Width of the feature map
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)

        # Pad if H/W is odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        H, W = H // 2, W // 2
        return x, H, W


class WindowAttention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 window_size: int = 7, 
                 heads: int = 8, 
                 dropout: float = 0.0) -> None:
        """
        Window-based Multi-Head Self-Attention module
        
        Parameters:
            dim: Dimension of the input features
            window_size: Size of the attention window
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 heads: int,
                 window_size: int, 
                 mlp_dim: int, 
                 dropout: float = 0.0) -> None:
        """
        Swin Transformer Block
        
        Parameters:
            dim: Dimension of the input features
            heads: Number of attention heads
            window_size: Size of the attention window
            mlp_dim: Dimension of the MLP hidden layer
            dropout: Dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, 
                 image_size: int = 224, 
                 patch_size: int = 4, 
                 in_chans: int = 3, 
                 dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.H = image_size // patch_size
        self.W = image_size // patch_size
        self.num_patches = self.H * self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Swin(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 dim: int = 96,
                 depths: List[int] = [2, 2, 6, 2],
                 heads: List[int] = [3, 6, 12, 24],
                 window_size: int = 7,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        """
        Swin Transformer model
        
        Parameters:
            image_size: Size of the input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_chans: Number of input channels
            num_classes: Number of output classes
            dim: Dimension of the embedding space
            depths: List of depths for each stage
            heads: List of number of attention heads for each stage
            window_size: Size of the attention window
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, dim)
        self.num_classes = num_classes

        # Build stages
        self.stages = nn.ModuleList()
        dim = dim
        H, W = self.patch_embed.H, self.patch_embed.W
        for i_stage, (depth, head) in enumerate(zip(depths, heads)):
            blocks = nn.ModuleList([SwinBlock(dim, head, window_size, int(dim * mlp_ratio), dropout) for _ in range(depth)])
            self.stages.append(blocks)
            if i_stage < len(depths) - 1:
                self.stages.append(PatchMerging(dim, dim * 2))
                dim *= 2
                H, W = H // 2, W // 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)
        for stage in self.stages:
            if isinstance(stage, nn.ModuleList):
                for blk in stage:
                    x = blk(x)
            else:
                x, H, W = stage(x, H, W)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits

    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer = None,
            criterion: nn.Module = None,
            number_of_epochs: int = 5,
            verbose: bool = False) -> None:
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(number_of_epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader.dataset)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{number_of_epochs}], Loss: {avg_loss:.4f}")

    def predict(self, test_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        self.eval()
        all_preds = []
        with torch.no_grad():
            for images in test_loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]

                outputs = self(images)
                pred = outputs.argmax(dim=1)
                all_preds.append(pred)

        return torch.cat(all_preds)
    
    def __str__(self) -> str:
        return "Shifted Window Transformer (Swin)"
