# This script tests various custom autoencoder models
# The results are under print function calls and in data/img/result_autoencoder.png in case you dont want to run the code

import os
import sys
from typing import Tuple 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Importing the custom models
from models.deep_learning.autoencoder import Autoencoder
from models.deep_learning.autoencoder.AE import VanillaAE
from models.deep_learning.autoencoder.RegularizedAE import RegularizedAE
from models.deep_learning.autoencoder.ConvolutionalAE import ConvolutionalAE
from models.deep_learning.autoencoder.VAE import VAE
from models.deep_learning.autoencoder.AAE import AAE
from models.deep_learning.autoencoder.VQVAE import VQVAE
from models.deep_learning.autoencoder.MAE import MAE

# Function to load dSprites dataset
def dsprites_dataloader(size: int, 
                        latents_sizes: np.ndarray, 
                        latents_bases: np.ndarray, 
                        imgs: np.ndarray,
                        batch_size: int = 128, 
                        shuffle: bool = True) -> DataLoader:
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    indices = np.dot(samples, latents_bases).astype(int)
    
    data_tensor = torch.from_numpy(imgs[indices]).float()
    data_tensor = data_tensor.unsqueeze(1)
    
    dataset = TensorDataset(data_tensor, data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

# Function to evaluate model performance
def evaluate_performance(model: Autoencoder, 
                         dataloader: DataLoader) -> Tuple[float, float]:
    model.encoder.eval()
    model.decoder.eval()
    
    total_mse = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for imgs, _ in dataloader:
            reconstructed = model.reconstruct(imgs)
            if isinstance(reconstructed, (list, tuple)):
                reconstructed = reconstructed[0]
            
            mse_batch = F.mse_loss(reconstructed, imgs)
            psnr_batch = 10 * torch.log10(1.0 / (mse_batch + 1e-10))
            
            total_mse += mse_batch.item()
            total_psnr += psnr_batch.item()
            num_batches += 1
            
    avg_mse = total_mse / num_batches
    avg_psnr = total_psnr / num_batches
    
    return avg_mse, avg_psnr

if __name__ == "__main__":
    # === Load Dataset === 
    # Load dSprites dataset
    # https://github.com/google-deepmind/dsprites-dataset
    dsprites_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", encoding="latin1", allow_pickle=True)
    dsprites_imgs = dsprites_zip["imgs"]
    dsprites_metadata = dsprites_zip["metadata"][()]
    dsprites_latents_sizes = dsprites_metadata["latents_sizes"]
    dsprites_latents_bases = np.concatenate((dsprites_latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))

    train_dsprites_loader = dsprites_dataloader(size=60000,
                                                latents_sizes=dsprites_latents_sizes,
                                                latents_bases=dsprites_latents_bases,
                                                imgs=dsprites_imgs,
                                                batch_size=128,
                                                shuffle=True)
    test_dsprites_loader = dsprites_dataloader(size=5000,
                                               latents_sizes=dsprites_latents_sizes,
                                               latents_bases=dsprites_latents_bases,
                                               imgs=dsprites_imgs,
                                               batch_size=128,
                                               shuffle=False)
    # ====================

    
    # === Test Autoencoder === 
    models = {
        "AE": VanillaAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_dims=[512, 256]),
        "Sparse AE": RegularizedAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_dims=[512, 256], reg_type="sparse", reg_coeff=1e-4),
        "Denoising AE": RegularizedAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_dims=[512, 256], reg_type="denoising", noise_factor=0.5),
        "Contractive AE": RegularizedAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_dims=[512, 256], reg_type="contractive", reg_coeff=1e-4),
        "Convolutional AE": ConvolutionalAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_channels=[32, 64, 128]),
        "VAE": VAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_dims=[512, 256], kld_weight=0.5),
        "AAE" : AAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=2e-4, number_of_epochs=20, hidden_dims=[512, 256]),
        "VQ-VAE": VQVAE(latent_dim=10, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20, hidden_channels=[32, 64]),
        "MAE": MAE(embed_dim=12, input_shape=(1, 64, 64), learn_rate=1e-3, number_of_epochs=20)
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")

        model.fit(train_loader=train_dsprites_loader, verbose=True)

        mse_score, psnr_score = evaluate_performance(model=model, dataloader=test_dsprites_loader)
        print(f"Reconstruction Error (MSE): {mse_score:.6f}")
        print(f"PSNR: {psnr_score:.2f} dB")

    """
    ==============================================================
    AE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.004752
    Epoch [10/20] | Loss: 0.003776
    Epoch [15/20] | Loss: 0.003354
    Epoch [20/20] | Loss: 0.003091
    Reconstruction Error (MSE): 0.002873
    PSNR: 25.43 dB
        
    ==============================================================
    Sparse AE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.004897
    Epoch [10/20] | Loss: 0.003945
    Epoch [15/20] | Loss: 0.003504
    Epoch [20/20] | Loss: 0.003248
    Reconstruction Error (MSE): 0.003043
    PSNR: 25.18 dB
    
    ==============================================================
    Denoising AE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.006752
    Epoch [10/20] | Loss: 0.005933
    Epoch [15/20] | Loss: 0.005467
    Epoch [20/20] | Loss: 0.005157
    Reconstruction Error (MSE): 0.006746
    PSNR: 21.72 dB
    
    ==============================================================
    Contractive AE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.004826
    Epoch [10/20] | Loss: 0.003875
    Epoch [15/20] | Loss: 0.003455
    Epoch [20/20] | Loss: 0.003190
    Reconstruction Error (MSE): 0.003038
    PSNR: 25.19 dB

    ==============================================================
    Convolutional AE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.008660
    Epoch [10/20] | Loss: 0.002933
    Epoch [15/20] | Loss: 0.002243
    Epoch [20/20] | Loss: 0.002032
    Reconstruction Error (MSE): 0.006362
    PSNR: 21.97 dB

    ==============================================================
    VAE Result
    ==============================================================
    Epoch [5/20] | Total Loss: 34.0436 (Recon: 25.9195, KLD: 16.2481)
    Epoch [10/20] | Total Loss: 31.5239 (Recon: 23.0656, KLD: 16.9165)
    Epoch [15/20] | Total Loss: 30.2002 (Recon: 21.6011, KLD: 17.1982)
    Epoch [20/20] | Total Loss: 29.4784 (Recon: 20.7964, KLD: 17.3640)
    Reconstruction Error (MSE): 0.004904
    PSNR: 23.10 dB

    ==============================================================
    AAE Result
    ==============================================================
    Epoch [5/20] | Recon: 0.0075 | Disc: 0.6939 | Gen: 0.6966
    Epoch [10/20] | Recon: 0.0068 | Disc: 0.6939 | Gen: 0.6916
    Epoch [15/20] | Recon: 0.0063 | Disc: 0.6935 | Gen: 0.6937
    Epoch [20/20] | Recon: 0.0062 | Disc: 0.6941 | Gen: 0.6934
    Reconstruction Error (MSE): 0.006883
    PSNR: 21.63 dB

    ==============================================================
    VQ-VAE Result
    ==============================================================
    Epoch [5/20] | Total Loss: 0.0035 (Recon: 0.0007, VQ: 0.0027)
    Epoch [10/20] | Total Loss: 0.0017 (Recon: 0.0002, VQ: 0.0014)
    Epoch [15/20] | Total Loss: 0.0010 (Recon: 0.0001, VQ: 0.0009)
    Epoch [20/20] | Total Loss: 0.0007 (Recon: 0.0001, VQ: 0.0006)
    Reconstruction Error (MSE): 0.000064
    PSNR: 42.04 dB

    ==============================================================
    MAE Result
    ==============================================================
    Epoch [5/20] | Loss: 0.023355
    Epoch [10/20] | Loss: 0.010881
    Epoch [15/20] | Loss: 0.009457
    Epoch [20/20] | Loss: 0.008664
    Reconstruction Error (MSE): 0.009084
    PSNR: 20.44 dB
    """