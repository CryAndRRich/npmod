# This script tests various Vision Transformer models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split

from models.deep_learning.transformer.vit.ViT import ViT
from models.deep_learning.transformer.vit.DeiT import DeiT
from models.deep_learning.transformer.vit.Swin import Swin

if __name__ == "__main__":
    # === Load Dataset === 
    # Load Oxford-IIIT-Pet dataset
    transform = Compose([Resize((224, 224)), ToTensor()])
    dataset = OxfordIIITPet(root=".", download=True, transform=transform)

    train_split = int(0.8 * len(dataset))
    train_oxford_idx, test_oxford_idx = random_split(dataset, [train_split, len(dataset) - train_split])

    train_oxford_loader = DataLoader(train_oxford_idx, batch_size=32, shuffle=True)
    test_oxford_loader = DataLoader(test_oxford_idx, batch_size=32, shuffle=True)

    y_oxford_test = []
    for _, targets in test_oxford_loader:
        y_oxford_test.append(targets)

    y_oxford_test = torch.cat(y_oxford_test)
    # ====================


    # === Test Vision Transformer ===
    models = {
        "ViT": ViT(
            image_size=224,
            patch_size=16,     
            num_classes=37,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768,
        ),
        "DeiT": DeiT(
            image_size=224,
            patch_size=16,     
            num_classes=37,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768
        ),
        "Swin": Swin(
            image_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=37,
            dim=64,
            depths=[2,2,2,2],
            heads=[2,4,8,16],
            window_size=7
        )
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")

        model.fit(train_loader=train_oxford_loader, number_of_epochs=50, verbose=True)

        preds = model.predict(test_loader=test_oxford_loader)
        accuracy = (preds == y_oxford_test).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    """
    ==============================================================
    ViT Result
    ==============================================================
    Epoch [10/50], Loss: 0.0539
    Epoch [20/50], Loss: 0.0481
    Epoch [30/50], Loss: 0.0418
    Epoch [40/50], Loss: 0.0320
    Epoch [50/50], Loss: 0.0173
    Test Accuracy: 2.72%

    ==============================================================
    DeiT Result
    ==============================================================
    Epoch [10/50], Loss: 0.0518
    Epoch [20/50], Loss: 0.0387
    Epoch [30/50], Loss: 0.0153
    Epoch [40/50], Loss: 0.0029
    Epoch [50/50], Loss: 0.0000
    Test Accuracy: 2.45%

    ==============================================================
    Swin Result
    ==============================================================
    Epoch [10/50], Loss: 0.1155
    Epoch [20/50], Loss: 0.1148
    Epoch [30/50], Loss: 0.1144
    Epoch [40/50], Loss: 0.1139
    Epoch [50/50], Loss: 0.1137
    Test Accuracy: 3.26%
    """