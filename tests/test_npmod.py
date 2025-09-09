# This script tests the custom numpy-based neural network modules against PyTorch implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Importing the custom modules
import npmod.nn as npm


# Function to train and evaluate PyTorch models
def train_nn(model: nn.Module, 
             loader: DataLoader, 
             optimizer: optim.Optimizer, 
             criterion: nn.Module, 
             device: torch.device) -> tuple:
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate_nn(model: nn.Module, 
                loader: DataLoader, 
                criterion: nn.Module, 
                device: torch.device) -> tuple:
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc
# ====================    

# Function to train and evaluate npmod models
def train_npm(model: npm.Sequential, 
              loader: DataLoader, 
              optimizer: npm.Optimizer, 
              criterion: npm.Loss) -> tuple:
    total_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for images_t, labels_t in loader:
        images = images_t.numpy().astype(np.float32)  
        labels = labels_t.numpy().astype(np.int64)    

        outputs = model(images)                    
        loss = criterion(outputs, labels)          
        model.backward(criterion)
        optimizer.step()

        N = images.shape[0]
        total_loss += loss * N
        preds = np.argmax(outputs, axis=1)
        correct += int((preds == labels).sum())
        total += N

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate_npm(model: npm.Sequential, 
                 loader: DataLoader, 
                 criterion: npm.Loss) -> tuple:
    total_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    for images_t, labels_t in loader:
        images = images_t.numpy().astype(np.float32)
        labels = labels_t.numpy().astype(np.int64)

        outputs = model(images)
        loss = criterion(outputs, labels)

        N = images.shape[0]
        total_loss += loss * N
        preds = np.argmax(outputs, axis=1)
        correct += int((preds == labels).sum())
        total += N

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc
# ====================

# Function to run training and evaluation for both PyTorch and npmod models
def run_model(model: nn.Module | npm.Sequential,
              train_loader: DataLoader, 
              test_loader: DataLoader, 
              optimizer: optim.Optimizer | npm.Optimizer, 
              criterion: nn.Module | npm.Loss,
              num_epochs: int = 10,
              is_npm: bool = False, 
              device: torch.device = None) -> None:
    
    for epoch in range(num_epochs):
        if is_npm:
            train_loss, train_acc = train_npm(model, train_loader, optimizer, criterion)
            test_loss, test_acc = evaluate_npm(model, test_loader, criterion)
        else:
            train_loss, train_acc = train_nn(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate_nn(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
# ====================


if __name__ == "__main__":
    # === Load all datasets ===
    # Load MNIST dataset 
    batch_mnist_size = 64
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform_mnist, download=True)
    test_mnist_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist, download=True)

    train_mnist_loader = DataLoader(train_mnist_dataset, batch_size=batch_mnist_size, shuffle=True, num_workers=0)
    test_mnist_loader = DataLoader(test_mnist_dataset, batch_size=batch_mnist_size, shuffle=False, num_workers=0)
    
    # Load satellite dataset
    # https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
    data_satellite_dir = "/data/satellite"
    batch_satellite_size = 64
    img_satellite_size = 224 
    transform_satellite = transforms.Compose([
        transforms.Resize((img_satellite_size, img_satellite_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    satellite_dataset = datasets.ImageFolder(root=data_satellite_dir, transform=transform_satellite)

    train_satellite_size = int(0.8 * len(satellite_dataset))
    test_satellite_size = len(satellite_dataset) - train_satellite_size

    train_satellite_dataset, test_satellite_dataset = random_split(
        satellite_dataset, [train_satellite_size, test_satellite_size]
    )
    
    train_satellite_loader = DataLoader(train_satellite_dataset, batch_size=batch_satellite_size, shuffle=True, num_workers=0)
    test_satellite_loader = DataLoader(test_satellite_dataset, batch_size=batch_satellite_size, shuffle=False, num_workers=0)
    # ====================
    
    # === Test Models === #
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=28*28, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=128, out_features=10)
    ).to(device1)
    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)

    run_model(model1, train_mnist_loader, test_mnist_loader, optimizer1, criterion1, num_epochs=10, is_npm=False, device=device1)
    """
    Epoch [1/10] Train Loss: 0.2632, Train Acc: 0.9223 | Test Loss: 0.1344, Test Acc: 0.9586
    Epoch [2/10] Train Loss: 0.2465, Train Acc: 0.9268 | Test Loss: 0.1243, Test Acc: 0.9619
    Epoch [3/10] Train Loss: 0.2389, Train Acc: 0.9298 | Test Loss: 0.1228, Test Acc: 0.9628
    Epoch [4/10] Train Loss: 0.2316, Train Acc: 0.9321 | Test Loss: 0.1244, Test Acc: 0.9634
    Epoch [5/10] Train Loss: 0.2296, Train Acc: 0.9318 | Test Loss: 0.1235, Test Acc: 0.9631
    Epoch [6/10] Train Loss: 0.2240, Train Acc: 0.9335 | Test Loss: 0.1136, Test Acc: 0.9654
    Epoch [7/10] Train Loss: 0.2202, Train Acc: 0.9356 | Test Loss: 0.1134, Test Acc: 0.9651
    Epoch [8/10] Train Loss: 0.2101, Train Acc: 0.9380 | Test Loss: 0.1074, Test Acc: 0.9676
    Epoch [9/10] Train Loss: 0.2075, Train Acc: 0.9391 | Test Loss: 0.1188, Test Acc: 0.9639
    Epoch [10/10] Train Loss: 0.2052, Train Acc: 0.9393 | Test Loss: 0.1158, Test Acc: 0.9668
    """

    model2 = npm.Sequential(layers = [
        npm.Flatten(), 
        npm.Linear(in_features=28*28, out_features=256),   
        npm.ReLU(),
        npm.Dropout(keep_prob=0.5),
        npm.Linear(in_features=256, out_features=128),
        npm.ReLU(),
        npm.Dropout(keep_prob=0.5),
        npm.Linear(in_features=128, out_features=10)
    ])
    criterion2 = npm.CE()
    optimizer2 = npm.Adam(model2.get_layers(), learn_rate=1e-3)
    run_model(model2, train_mnist_loader, test_mnist_loader, optimizer2, criterion2, num_epochs=10, is_npm=True)
    """
    Epoch [1/10] Train Loss: 0.6352, Train Acc: 0.7967 | Test Loss: 0.2574, Test Acc: 0.9186
    Epoch [2/10] Train Loss: 0.3606, Train Acc: 0.8930 | Test Loss: 0.1952, Test Acc: 0.9386
    Epoch [3/10] Train Loss: 0.3154, Train Acc: 0.9077 | Test Loss: 0.1674, Test Acc: 0.9476
    Epoch [4/10] Train Loss: 0.2866, Train Acc: 0.9157 | Test Loss: 0.1547, Test Acc: 0.9516
    Epoch [5/10] Train Loss: 0.2739, Train Acc: 0.9173 | Test Loss: 0.1429, Test Acc: 0.9565
    Epoch [6/10] Train Loss: 0.2548, Train Acc: 0.9249 | Test Loss: 0.1359, Test Acc: 0.9582
    Epoch [7/10] Train Loss: 0.2473, Train Acc: 0.9257 | Test Loss: 0.1320, Test Acc: 0.9598
    Epoch [8/10] Train Loss: 0.2435, Train Acc: 0.9274 | Test Loss: 0.1319, Test Acc: 0.9604
    Epoch [9/10] Train Loss: 0.2398, Train Acc: 0.9281 | Test Loss: 0.1290, Test Acc: 0.9603
    Epoch [10/10] Train Loss: 0.2318, Train Acc: 0.9307 | Test Loss: 0.1156, Test Acc: 0.9646
    """
    # ====================

    # === Test Models === #
    device3 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model3 = nn.Sequential(
        nn.Conv2d(in_channels=1, 
                  out_channels=8, 
                  kernel_size=(3, 3), 
                  stride=(1, 1), 
                  padding=(1, 1)),
        nn.MaxPool2d(kernel_size=2, 
                     stride=2, 
                     padding=0, 
                     dilation=1, 
                     ceil_mode=False),
        nn.Flatten(),
        nn.Linear(in_features=1568, out_features=10, bias=True)
    ).to(device3)
    criterion3 = nn.CrossEntropyLoss()
    optimizer3 = optim.Adam(model3.parameters(), lr=1e-3)

    run_model(model3, train_mnist_loader, test_mnist_loader, optimizer3, criterion3, num_epochs=3, is_npm=False, device=device3)
    """
    Epoch [1/3] Train Loss: 0.3645, Train Acc: 0.8942 | Test Loss: 0.2105, Test Acc: 0.9386
    Epoch [2/3] Train Loss: 0.1831, Train Acc: 0.9478 | Test Loss: 0.1440, Test Acc: 0.9570
    Epoch [3/3] Train Loss: 0.1313, Train Acc: 0.9618 | Test Loss: 0.1157, Test Acc: 0.9652
    """

    model4 = npm.Sequential(layers = [
        npm.Conv(in_channels=1, 
                 out_channels=8, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1),
        npm.Pooling(kernel_size=2, 
                    stride=2,
                    padding=0,
                    mode="max"),
        npm.Flatten(),
        npm.Linear(in_features=1568, out_features=10)
    ])
    criterion4 = npm.CE()
    optimizer4 = npm.Adam(model4.get_layers(), learn_rate=1e-3)

    run_model(model4, train_mnist_loader, test_mnist_loader, optimizer4, criterion4, num_epochs=3, is_npm=True)
    """
    Epoch [1/3] Train Loss: 0.4544, Train Acc: 0.8651 | Test Loss: 0.3172, Test Acc: 0.9035
    Epoch [2/3] Train Loss: 0.3314, Train Acc: 0.9019 | Test Loss: 0.3052, Test Acc: 0.9084
    Epoch [3/3] Train Loss: 0.3040, Train Acc: 0.9103 | Test Loss: 0.2631, Test Acc: 0.9247
    """
    # ====================

    # === Test Models === #
    device5 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model5 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=3 * 224 * 224, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=256, out_features=4)
    ).to(device5)
    criterion5 = nn.CrossEntropyLoss()
    optimizer5 = optim.Adam(model5.parameters(), lr=1e-2)

    run_model(model5, train_satellite_loader, test_satellite_loader, optimizer5, criterion5, num_epochs=10, is_npm=False, device=device5)
    """
    Epoch [1/10] Train Loss: 107.3521, Train Acc: 0.6745 | Test Loss: 19.1997, Test Acc: 0.8225
    Epoch [2/10] Train Loss: 43.0706, Train Acc: 0.7131 | Test Loss: 86.6513, Test Acc: 0.6131
    Epoch [3/10] Train Loss: 70.1641, Train Acc: 0.7116 | Test Loss: 88.7976, Test Acc: 0.6788
    Epoch [4/10] Train Loss: 42.0031, Train Acc: 0.7489 | Test Loss: 34.8812, Test Acc: 0.6699
    Epoch [5/10] Train Loss: 44.6281, Train Acc: 0.7140 | Test Loss: 33.8331, Test Acc: 0.7542
    Epoch [6/10] Train Loss: 35.1260, Train Acc: 0.7360 | Test Loss: 35.4396, Test Acc: 0.7223
    Epoch [7/10] Train Loss: 63.0253, Train Acc: 0.7271 | Test Loss: 42.6310, Test Acc: 0.7595
    Epoch [8/10] Train Loss: 79.6620, Train Acc: 0.7267 | Test Loss: 45.0698, Test Acc: 0.7338
    Epoch [9/10] Train Loss: 106.9573, Train Acc: 0.7151 | Test Loss: 93.5938, Test Acc: 0.7205
    Epoch [10/10] Train Loss: 87.9645, Train Acc: 0.7436 | Test Loss: 41.9694, Test Acc: 0.7959
    """

    model6 = npm.Sequential(layers = [
        npm.Flatten(), 
        npm.Linear(in_features=3 * 224 * 224, out_features=256),   
        npm.ReLU(),
        npm.Dropout(keep_prob=0.3),
        npm.Linear(in_features=256, out_features=4)
    ])
    criterion6 = npm.CE()
    optimizer6 = npm.Adam(model6.get_layers(), learn_rate=1e-2)

    run_model(model6, train_satellite_loader, test_satellite_loader, optimizer6, criterion6, num_epochs=10, is_npm=True)
    """
    Epoch [1/10] Train Loss: 7.3912, Train Acc: 0.6168 | Test Loss: 7.4447, Test Acc: 0.6362
    Epoch [2/10] Train Loss: 6.8628, Train Acc: 0.6565 | Test Loss: 5.1688, Test Acc: 0.7445
    Epoch [3/10] Train Loss: 6.3964, Train Acc: 0.6723 | Test Loss: 5.1038, Test Acc: 0.7400
    Epoch [4/10] Train Loss: 6.0588, Train Acc: 0.6858 | Test Loss: 3.4560, Test Acc: 0.7338
    Epoch [5/10] Train Loss: 6.2425, Train Acc: 0.6734 | Test Loss: 5.2353, Test Acc: 0.7374
    Epoch [6/10] Train Loss: 6.0383, Train Acc: 0.6650 | Test Loss: 2.9739, Test Acc: 0.8092
    Epoch [7/10] Train Loss: 6.0023, Train Acc: 0.6823 | Test Loss: 3.6687, Test Acc: 0.7995
    Epoch [8/10] Train Loss: 5.6536, Train Acc: 0.6867 | Test Loss: 3.3356, Test Acc: 0.7950
    Epoch [9/10] Train Loss: 5.9417, Train Acc: 0.6876 | Test Loss: 4.5556, Test Acc: 0.7604
    Epoch [10/10] Train Loss: 5.7813, Train Acc: 0.6770 | Test Loss: 4.9332, Test Acc: 0.7409
    """
    # ====================