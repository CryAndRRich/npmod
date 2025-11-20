from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from ..autoencoder import Autoencoder

class PatchEmbed(nn.Module):
    def __init__(self, 
                 img_size: int = 64, 
                 patch_size: int = 4, 
                 in_chans: int = 1, 
                 embed_dim: int = 128) -> None:
        """
        Patch Embedding Layer
        
        Parameters:
            img_size: Size of the input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_chans: Number of input channels
            embed_dim: Dimension of the embedding space
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels=in_chans, 
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 drop: float = 0.0) -> None:
        """
        Transformer Block
        
        Parameters:
            dim: Dimension of the input and output
            num_heads: Number of attention heads
            mlp_ratio: Ratio of the MLP hidden dimension to input dimension
            drop: Dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, 
                                          num_heads=num_heads, 
                                          dropout=drop, 
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(normalized_shape=dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(in_features=int(dim * mlp_ratio), out_features=dim),
            nn.Dropout(p=drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 num_patches: int, 
                 embed_dim: int, 
                 depth: int, 
                 num_heads: int) -> None:
        """
        MAE Encoder
        
        Parameters:
            num_patches: Total number of patches in the input
            embed_dim: Dimension of the embedding space
            depth: Number of Transformer blocks
            num_heads: Number of attention heads
        """
        super().__init__()
        self.pos_embed = nn.Parameter(data=torch.zeros(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, 
                x: torch.Tensor, 
                mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.pos_embed
        
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)
        
        return x_masked, ids_restore, ids_keep

class Decoder(nn.Module):
    def __init__(self, 
                 num_patches: int, 
                 enc_dim: int, 
                 dec_dim: int, 
                 depth: int, 
                 num_heads: int, 
                 pixel_values_per_patch: int) -> None:
        """
        MAE Decoder
        
        Parameters:
            num_patches: Total number of patches in the input
            enc_dim: Dimension of the encoder embedding space
            dec_dim: Dimension of the decoder embedding space
            depth: Number of Transformer blocks
            num_heads: Number of attention heads
            pixel_values_per_patch: Number of pixel values per patch
        """
        super().__init__()
        self.decoder_embed = nn.Linear(in_features=enc_dim, 
                                       out_features=dec_dim, 
                                       bias=True)
        self.mask_token = nn.Parameter(data=torch.zeros(1, 1, dec_dim))
        
        self.decoder_pos_embed = nn.Parameter(data=torch.zeros(1, num_patches, dec_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dec_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(normalized_shape=dec_dim)
        
        self.decoder_pred = nn.Linear(in_features=dec_dim, 
                                      out_features=pixel_values_per_patch, 
                                      bias=True)

    def forward(self, 
                z: torch.Tensor, 
                ids_restore: torch.Tensor) -> torch.Tensor:
        z = self.decoder_embed(z)
        mask_tokens = self.mask_token.repeat(z.shape[0], ids_restore.shape[1] - z.shape[1], 1)
        
        z0 = torch.cat([z, mask_tokens], dim=1)  
        z0 = torch.gather(z0, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, z.shape[2]))

        z = z0 + self.decoder_pos_embed
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        z = self.decoder_pred(z)
        return z

class MAE(Autoencoder):
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 patch_size: int = 4,
                 embed_dim: int = 128,
                 decoder_dim: int = 64,
                 encoder_depth: int = 4,
                 decoder_depth: int = 2,
                 num_heads: int = 4,
                 mask_ratio: float = 0.75,
                 learn_rate: float = 1e-3,
                 number_of_epochs: int = 50) -> None:
        """
        Masked Autoencoder
        
        Parameters:
            input_shape: Shape of the input data (C, H, W)
            patch_size: Size of each patch (assumed square)
            embed_dim: Dimension of the embedding space
            decoder_dim: Dimension of the decoder embedding space
            encoder_depth: Number of Transformer blocks in the encoder
            decoder_depth: Number of Transformer blocks in the decoder
            num_heads: Number of attention heads
            mask_ratio: Ratio of patches to mask during training
            learn_rate: Learning rate for the optimizer
            number_of_epochs: Number of training iterations
        """
        super().__init__(
            latent_dim=embed_dim, 
            input_shape=input_shape, 
            learn_rate=learn_rate, 
            number_of_epochs=number_of_epochs
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.enc_depth = encoder_depth
        self.dec_depth = decoder_depth
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        
        C, H, W = input_shape
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.pixel_values_per_patch = patch_size * patch_size * C

    def init_network(self) -> None:
        self.patch_embed = PatchEmbed(self.input_shape[1], self.patch_size, self.input_shape[0], self.embed_dim)
        self.encoder = Encoder(self.num_patches, self.embed_dim, self.enc_depth, self.num_heads)
        self.decoder = Decoder(self.num_patches, self.embed_dim, self.decoder_dim, self.dec_depth, self.num_heads, self.pixel_values_per_patch)

        self.patch_embed.apply(self.init_weights)
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        params = list(self.patch_embed.parameters()) + \
                 list(self.encoder.parameters()) + \
                 list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=self.learn_rate, weight_decay=0.05)
        self.criterion = nn.MSELoss()

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3 if imgs.shape[1]==3 else 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * imgs.shape[1]))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1 if self.input_shape[0]==1 else 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1 if self.input_shape[0]==1 else 3, h * p, h * p))
        return imgs

    def fit(self, 
            dataloader: torch.utils.data.DataLoader, 
            verbose: bool = False) -> None:
        self.init_network()
        self.patch_embed.train()
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(self.number_of_epochs):
            total_loss = 0.0
            for img, _ in dataloader:
                patches = self.patch_embed(img)
                
                latent, ids_restore, _ = self.encoder(patches, mask_ratio=self.mask_ratio)
                
                pred = self.decoder(latent, ids_restore)
                
                target = self.patchify(img)
                loss = self.criterion(pred, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] | Loss: {total_loss/len(dataloader):.6f}")

    def reconstruct(self, 
                    x: torch.Tensor, 
                    mask_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        self.patch_embed.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            patches = self.patch_embed(x)
            latent, ids_restore, ids_keep = self.encoder(patches, mask_ratio=mask_ratio)
            pred_patches = self.decoder(latent, ids_restore)
            reconstruction = self.unpatchify(pred_patches)
            
            target_patches = self.patchify(x) 
            B, L, D_pix = target_patches.shape 
            
            mask_pixel_tokens = torch.zeros(B, L - ids_keep.shape[1], D_pix, device=x.device)
            visible_pixel_patches = torch.gather(target_patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D_pix))
            mixed_pixels = torch.cat([visible_pixel_patches, mask_pixel_tokens], dim=1)
            mixed_pixels = torch.gather(mixed_pixels, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_pix))
            masked_image = self.unpatchify(mixed_pixels)

        return reconstruction, masked_image
    
    def __str__(self) -> str:
        return "Masked Autoencoder (MAE)"