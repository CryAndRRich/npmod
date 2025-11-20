import torch
import torch.nn as nn
from ..vit import *

class DeiTBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 mlp_dim: int, 
                 dropout: float = 0.0, 
                 init_scale: float = 1e-4) -> None:
        """
        DeiT Encoder Block with optional LayerScale

        Parameters:
            dim: Dimension of the input features
            heads: Number of attention heads
            mlp_dim: Dimension of the MLP hidden layer
            dropout: Dropout rate
            init_scale: Initial scale for LayerScale
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.gamma_1 = nn.Parameter(init_scale * torch.ones(dim))
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        self.gamma_2 = nn.Parameter(init_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class DeiT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 37,
                 dim: int = 192,
                 depth: int = 6,
                 heads: int = 3,
                 mlp_dim: int = 768,
                 dropout: float = 0.0,
                 emb_dropout: float = 0.0) -> None:
        """
        Data-efficient Image Transformer (DeiT) model

        Parameters:
            image_size: Size of the input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_chans: Number of input channels
            num_classes: Number of output classes
            dim: Dimension of the embedding space
            depth: Number of Transformer blocks
            heads: Number of attention heads
            mlp_dim: Dimension of the MLP hidden layer
            dropout: Dropout rate
            emb_dropout: Embedding dropout rate
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([DeiTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]
        logits = self.head(cls)
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
                print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.4f}")

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
        return "Data-efficient Image Transformer (DeiT)"