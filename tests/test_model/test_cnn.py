# This script tests various CNN models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.model_selection import train_test_split

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Importing the custom models
from models.deep_learning.cnn.LeNet import LeNet
from models.deep_learning.cnn.AlexNet import AlexNet
from models.deep_learning.cnn.NiN import NiN
from models.deep_learning.cnn.VGG import VGG
from models.deep_learning.cnn.GoogLeNet import GoogLeNet
from models.deep_learning.cnn.ResNet import ResNet
from models.deep_learning.cnn.SqueezeNet import SqueezeNet
from models.deep_learning.cnn.ResNeXt import ResNeXt
from models.deep_learning.cnn.Xception import Xception
from models.deep_learning.cnn.DenseNet import DenseNet
from models.deep_learning.cnn.MobileNet import MobileNet
from models.deep_learning.cnn.NASNet import NASNet
from models.deep_learning.cnn.ShuffleNet import ShuffleNet
from models.deep_learning.cnn.EfficientNet import EfficientNet
from models.deep_learning.cnn.RegNet import RegNet
from models.deep_learning.cnn.GhostNet import GhostNet
from models.deep_learning.cnn.MicroNet import MicroNet

if __name__ == "__main__":
    # === Load Dataset === 
    # Load MNIST dataset
    transform_mnist = transforms.ToTensor()
    train_mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_mnist)
    test_mnist_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_mnist)

    train_mnist_loader = DataLoader(train_mnist_dataset, batch_size=128, shuffle=True)
    test_mnist_loader = DataLoader(test_mnist_dataset, batch_size=128, shuffle=False)

    mnist_labels = []

    for _, labels in test_mnist_loader:  
        mnist_labels.append(labels)

    mnist_labels = torch.cat(mnist_labels, dim=0)

    # Load Tom and Jerry dataset
    # https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification
    tom_jerry_dir = r"D:\Project\npmod\data\tom_and_jerry.zip"
    transform_tom_jerry = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    full_tom_jerry_dataset = datasets.ImageFolder(root=tom_jerry_dir, transform=transform_tom_jerry)
    tom_jerry_classes = len(full_tom_jerry_dataset.classes)

    train_tom_jerry_indices, test_tom_jerry_indices = train_test_split(
        list(range(len(full_tom_jerry_dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_tom_jerry_dataset = Subset(full_tom_jerry_dataset, train_tom_jerry_indices)
    test_tom_jerry_dataset  = Subset(full_tom_jerry_dataset, test_tom_jerry_indices)

    train_tom_jerry_loader = DataLoader(train_tom_jerry_dataset, batch_size=64, shuffle=True)
    test_tom_jerry_loader  = DataLoader(test_tom_jerry_dataset, batch_size=64, shuffle=False)

    tom_jerry_labels = []
    for _, labels in test_tom_jerry_loader:
        tom_jerry_labels.append(labels)

    tom_jerry_labels = torch.cat(tom_jerry_labels, dim=0)
    # ====================


    # === Test CNN === #
    models = {
        "LeNet": LeNet(learn_rate=0.01, number_of_epochs=5),

        "AlexNet": AlexNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "NiN": NiN(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "VGG-16": VGG(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "GoogLeNet": GoogLeNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "ResNet-34": ResNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "SqueezeNet-1.1": SqueezeNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "ResNeXt-50": ResNeXt(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "DenseNet-121": DenseNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "MobileNetV1": MobileNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "NASNet": NASNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "ShuffleNetV1": ShuffleNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "EfficientNet-Lite": EfficientNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "RegNet-200MF": RegNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "GhostNet": GhostNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "MicroNet": MicroNet(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes),
        "Xception": Xception(learn_rate=0.01, number_of_epochs=5, out_channels=tom_jerry_classes)
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")
        
        if name == "LeNet":
            model.fit(train_mnist_loader, verbose=True)
            preds = model.predict(test_mnist_loader)
            print("Accuracy: ", (preds == mnist_labels).sum().item() / mnist_labels.size(0))
        else:
            model.fit(train_tom_jerry_loader, verbose=True)
            preds = model.predict(test_tom_jerry_loader)
            print("Accuracy: ", (preds == tom_jerry_labels).sum().item() / tom_jerry_labels.size(0))    
    
    """
    ==============================================================
    LeNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1098
    Epoch [2/5], Loss: 0.0483
    Epoch [3/5], Loss: 0.0374
    Epoch [4/5], Loss: 0.0310
    Epoch [5/5], Loss: 0.0270
    Accuracy:  0.9885
    """

    """
    ==============================================================
    AlexNet Result
    ==============================================================
    Epoch [1/5], Loss: 4.5504
    Epoch [2/5], Loss: 1.7464
    Epoch [3/5], Loss: 2.0272
    Epoch [4/5], Loss: 1.4630
    Epoch [5/5], Loss: 0.9794
    Accuracy:  0.5483576642335767

	==============================================================
    NiN Result
    ==============================================================
    Epoch [1/5], Loss: 1.1628
    Epoch [2/5], Loss: 0.9263
    Epoch [3/5], Loss: 0.8358
    Epoch [4/5], Loss: 0.7688
    Epoch [5/5], Loss: 0.6954
    Accuracy:  0.6779197080291971

    ==============================================================
    VGG Result
    ==============================================================
    Epoch [1/5], Loss: 4.0051
    Epoch [2/5], Loss: 3.3550
    Epoch [3/5], Loss: 3.3029
    Epoch [4/5], Loss: 3.0497
    Epoch [5/5], Loss: 3.2295
    Accuracy:  0.2782846715328467

    ==============================================================
    GoogLeNet Result
    ==============================================================
    Epoch [1/5], Loss: 1.6051
    Epoch [2/5], Loss: 1.1241
    Epoch [3/5], Loss: 0.9232
    Epoch [4/5], Loss: 0.8841
    Epoch [5/5], Loss: 0.7817
    Accuracy:  0.6268248175182481

    ==============================================================
    ResNet-34 Result
    ==============================================================
    Epoch [1/5], Loss: 2.5898
    Epoch [2/5], Loss: 1.3743
    Epoch [3/5], Loss: 1.3442
    Epoch [4/5], Loss: 1.3493
    Epoch [5/5], Loss: 1.3433
    Accuracy:  0.36313868613138683

    ==============================================================
    SqueezeNet-1.1 Result
    ==============================================================
    Epoch [1/5], Loss: 1.2108
    Epoch [2/5], Loss: 1.0025
    Epoch [3/5], Loss: 0.8989
    Epoch [4/5], Loss: 0.8407
    Epoch [5/5], Loss: 0.7747
    Accuracy:  0.5775547445255474

    ==============================================================
    ResNeXt Result
    ==============================================================
    Epoch [1/5], Loss: 3.3535
    Epoch [2/5], Loss: 1.4265
    Epoch [3/5], Loss: 1.2502
    Epoch [4/5], Loss: 1.1688
    Epoch [5/5], Loss: 1.1097
    Accuracy:  0.5018248175182481

    ==============================================================
    DenseNet-121 Result
    ==============================================================
    Epoch [1/5], Loss: 1.5450
    Epoch [2/5], Loss: 1.3419
    Epoch [3/5], Loss: 1.2775
    Epoch [4/5], Loss: 1.1625
    Epoch [5/5], Loss: 1.0193
    Accuracy:  0.47992700729927007

    ==============================================================
    MobileNetV1 Result
    ==============================================================
    Epoch [1/5], Loss: 1.4369
    Epoch [2/5], Loss: 0.8783
    Epoch [3/5], Loss: 0.7514
    Epoch [4/5], Loss: 0.6962
    Epoch [5/5], Loss: 0.6374
    Accuracy:  0.6578467153284672

    ==============================================================
    NASNet Result
    ==============================================================
    Epoch [1/5], Loss: 1.4869
    Epoch [2/5], Loss: 1.2321
    Epoch [3/5], Loss: 1.0991
    Epoch [4/5], Loss: 1.0445
    Epoch [5/5], Loss: 1.0051
    Accuracy:  0.3448905109489051

    ==============================================================
    ShuffleNetV1 Result
    ==============================================================
    Epoch [1/5], Loss: 2.7467
    Epoch [2/5], Loss: 0.9820
    Epoch [3/5], Loss: 0.9146
    Epoch [4/5], Loss: 0.7345
    Epoch [5/5], Loss: 0.6231
    Accuracy:  0.6815693430656934

    ==============================================================
    EfficientNet-Lite Result
    ==============================================================
    Epoch [1/5], Loss: 1.5644
    Epoch [2/5], Loss: 1.3513
    Epoch [3/5], Loss: 1.2870
    Epoch [4/5], Loss: 1.0057
    Epoch [5/5], Loss: 0.9127
    Accuracy:  0.5538321167883211

    ==============================================================
    RegNet-200MF Result
    ==============================================================
    Epoch [1/5], Loss: 2.0809
    Epoch [2/5], Loss: 1.1568
    Epoch [3/5], Loss: 0.9147
    Epoch [4/5], Loss: 0.6896
    Epoch [5/5], Loss: 0.6377
    Accuracy:  0.6578467153284672
        
    ==============================================================
    GhostNet Result
    ==============================================================
    Epoch [1/5], Loss: 1.4453
    Epoch [2/5], Loss: 0.8591
    Epoch [3/5], Loss: 0.8132
    Epoch [4/5], Loss: 0.6871
    Epoch [5/5], Loss: 0.6542
    Accuracy:  0.7208029197080292

    ==============================================================
    MicroNet Result
    ==============================================================
    Epoch [1/5], Loss: 1.2798
    Epoch [2/5], Loss: 0.8329
    Epoch [3/5], Loss: 0.7466
    Epoch [4/5], Loss: 0.6600
    Epoch [5/5], Loss: 0.5652
    Accuracy:  0.7974452554744526

    ==============================================================
    Xception Result
    ==============================================================
    Epoch [1/5], Loss: 1.8115
    Epoch [2/5], Loss: 1.1605
    Epoch [3/5], Loss: 1.0441
    Epoch [4/5], Loss: 1.0038
    Epoch [5/5], Loss: 0.9638
    Accuracy:  0.5465328467153284
    """