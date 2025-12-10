# This script tests custom DDIM
# The results are under print function calls in case you dont want to run the code

import os
import sys
from typing import Tuple 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.linalg

from torchvision import transforms
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Importing the custom models
from models.deep_learning.diffusion.unconditional import DDIM

# Loading FFHQ dataset
class FFHQDataset(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 transform: transforms.Compose = None, 
                 num_samples: int = 50000) -> None:
        self.root_dir = root_dir
        self.transform = transform
        all_images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {root_dir}")
            
        use_samples = min(len(all_images), num_samples)
        self.images = np.random.choice(all_images, use_samples, replace=False)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
# Function to evaluate model performance
def evaluate_performance(model: DDIM, 
                         dataloader: DataLoader,
                         max_images: int = 5000,
                         batch_size: int = 128,
                         device: str = "cuda") -> None:
    
    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()

    inception_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_features(images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            images = images.to(device)
            
            if images.shape[2] != 299 or images.shape[3] != 299:
                images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
            
            images = inception_normalize(images)
            
            feats = inception(images)
            
        return feats.cpu().numpy()

    print("Extracting Real Image Features...")

    real_features = []
    for imgs in tqdm(dataloader, desc="Real Images"):
        feats = get_features(imgs)
        real_features.append(feats)

    real_features = np.concatenate(real_features, axis=0)
    real_features = real_features[:max_images]

    print(f"Generating {max_images} Fake Images and Extracting Features...")

    fake_features = []
    num_fake = 0

    with torch.no_grad():
        pbar = tqdm(total=max_images, desc="Fake Images")
        while num_fake < max_images:
            current_batch_size = min(batch_size, max_images - num_fake)
            
            fake_batch = model.sample(
                n=current_batch_size, 
                ddim_timesteps=50, 
                eta=0.0
            )
            
            feats = get_features(fake_batch)
            fake_features.append(feats)
            
            num_fake += current_batch_size
            pbar.update(current_batch_size)
        pbar.close()

    fake_features = np.concatenate(fake_features, axis=0)
    def calculate_fid(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID Score: {fid_score:.4f}")

    def compute_pr(real_feats: np.ndarray, 
                   fake_feats: np.ndarray, 
                   k: int = 3) -> Tuple[float, float]:
        nn_real = NearestNeighbors(n_neighbors=k).fit(real_feats)
        
        dists_real, _ = nn_real.kneighbors(real_feats)
        radii_real = dists_real[:, -1]
        dists_fake_to_real, idxs_fake_to_real = nn_real.kneighbors(fake_feats, n_neighbors=1)
        
        matched_radii = radii_real[idxs_fake_to_real[:, 0]]
        
        precision = (dists_fake_to_real[:, 0] <= matched_radii).mean()

        nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_feats)
        
        dists_fake, _ = nn_fake.kneighbors(fake_feats)
        radii_fake = dists_fake[:, -1]
        
        dists_real_to_fake, idxs_real_to_fake = nn_fake.kneighbors(real_feats, n_neighbors=1)
        
        matched_radii_fake = radii_fake[idxs_real_to_fake[:, 0]]
        
        recall = (dists_real_to_fake[:, 0] <= matched_radii_fake).mean()

        return precision, recall

    precision, recall = compute_pr(real_features, fake_features, k=3)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
if __name__ == "__main__":
    # === Load Dataset === 
    # Load FFHQ dataset
    # https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
    transform_ffhq = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_ffhq_size = 128
    ffhq_dataset = FFHQDataset(root_dir="flickrfaceshq-dataset-ffhq", transform=transform_ffhq)
    ffhq_loader = DataLoader(ffhq_dataset, batch_size=batch_ffhq_size, shuffle=True, num_workers=4)
    # ====================

    # === Test DDIM === 
    models = DDIM(learn_rate=3e-4, number_of_epochs=40, device="cuda")

    print("Starting Training...")
    models.fit(ffhq_loader, verbose=True)
    evaluate_performance(model=models, 
                         dataloader=ffhq_loader, 
                         batch_size=batch_ffhq_size)

    """
    Starting Training...
    Epoch [5/40] | Avg Loss: 0.05282
    Epoch [10/40] | Avg Loss: 0.04463
    Epoch [15/40] | Avg Loss: 0.04134
    Epoch [20/40] | Avg Loss: 0.04033
    Epoch [25/40] | Avg Loss: 0.03897
    Epoch [30/40] | Avg Loss: 0.03819
    Epoch [35/40] | Avg Loss: 0.03768
    Epoch [40/40] | Avg Loss: 0.03723

    Extracting Real Image Features...
    Real Images: 100%|██████████| 40/40 [00:33<00:00,  2.33it/s]

    Generating 5000 Fake Images & Extracting Features...
    Fake Images: 100%|██████████| 5000/5000 [09:43<00:00,  8.57it/s]

    FID Score: 70.0098
    Precision: 0.0426
    Recall: 0.1780
    """