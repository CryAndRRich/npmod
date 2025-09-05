# This script tests various CNN models
# The results are under print function calls in case you dont want to run the code
# All CNN models are changed to fit MNIST dataset

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Importing the custom models
from models.deep_learning.cnn.LeNet import LeNet
from models.deep_learning.cnn.AlexNet import AlexNet
from models.deep_learning.cnn.NiN import NiN
from models.deep_learning.cnn.VGG import VGG
from models.deep_learning.cnn.GoogLeNet import GoogLeNet
from models.deep_learning.cnn.ResNet34 import ResNet34
from models.deep_learning.cnn.ResNet152 import ResNet152
from models.deep_learning.cnn.SqueezeNet import SqueezeNet
from models.deep_learning.cnn.ResNeXt import ResNeXt
from models.deep_learning.cnn.DenseNet import DenseNet
from models.deep_learning.cnn.WideResNet import WideResNet
from models.deep_learning.cnn.MobileNet import MobileNet
from models.deep_learning.cnn.NASNet import NASNet
from models.deep_learning.cnn.ShuffleNet import ShuffleNet
from models.deep_learning.cnn.EfficientNet import EfficientNet
from models.deep_learning.cnn.RegNet import RegNet
from models.deep_learning.cnn.GhostNet import GhostNet
from models.deep_learning.cnn.MicroNet import MicroNet

if __name__ == "__main__":
    # === Load Dataset === 
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # ====================


    # === Test CNN === #
    models = {
        "LeNet": LeNet(learn_rate=0.01, number_of_epochs=5),
        "AlexNet": AlexNet(learn_rate=0.01, number_of_epochs=5),
        "NiN": NiN(learn_rate=0.01, number_of_epochs=5),
        "VGG": VGG(learn_rate=0.01, number_of_epochs=5),
        "GoogLeNet": GoogLeNet(learn_rate=0.01, number_of_epochs=5),
        "ResNet34": ResNet34(learn_rate=0.01, number_of_epochs=5),
        "ResNet152": ResNet152(learn_rate=0.01, number_of_epochs=5),
        "SqueezeNet": SqueezeNet(learn_rate=0.01, number_of_epochs=5),
        "ResNeXt": ResNeXt(learn_rate=0.01, number_of_epochs=5),
        "DenseNet": DenseNet(learn_rate=0.01, number_of_epochs=5),
        "WideResNet": WideResNet(learn_rate=0.01, number_of_epochs=5),
        "MobileNet": MobileNet(learn_rate=0.01, number_of_epochs=5),
        "NASNet": NASNet(learn_rate=0.01, number_of_epochs=5),
        "ShuffleNet": ShuffleNet(learn_rate=0.01, number_of_epochs=5),
        "EfficientNet": EfficientNet(learn_rate=0.01, number_of_epochs=5),
        "RegNet": RegNet(learn_rate=0.01, number_of_epochs=5),
        "GhostNet": GhostNet(learn_rate=0.01, number_of_epochs=5),
        "MicroNet": MicroNet(learn_rate=0.01, number_of_epochs=5)
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")
        
        model.fit(train_loader, verbose=True)
        preds, labels = model.predict(test_loader)
        print("Accuracy: ", (preds == labels).sum().item() / labels.size(0))
    
    """
    ==============================================================
    LeNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1059
    Epoch [2/5], Loss: 0.0492
    Epoch [3/5], Loss: 0.0391
    Epoch [4/5], Loss: 0.0340
    Epoch [5/5], Loss: 0.0270
    Accuracy:  0.9909

    ==============================================================
    AlexNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1674
    Epoch [2/5], Loss: 0.0618
    Epoch [3/5], Loss: 0.0489
    Epoch [4/5], Loss: 0.0430
    Epoch [5/5], Loss: 0.0352
    Accuracy:  0.9893

	==============================================================
	NiN Result
	==============================================================
	Epoch [1/5], Loss: 0.3206
	Epoch [2/5], Loss: 0.0680
	Epoch [3/5], Loss: 0.0502
	Epoch [4/5], Loss: 0.0418
	Epoch [5/5], Loss: 0.0387
	Accuracy:  0.9874

    ==============================================================
    VGG Result
    ==============================================================
    Epoch [1/5], Loss: 0.2285
    Epoch [2/5], Loss: 0.0835
    Epoch [3/5], Loss: 0.0643
    Epoch [4/5], Loss: 0.0556
    Epoch [5/5], Loss: 0.0514
    Accuracy:  0.9887

    ==============================================================
    GoogLeNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1035
    Epoch [2/5], Loss: 0.0404
    Epoch [3/5], Loss: 0.0311
    Epoch [4/5], Loss: 0.0246
    Epoch [5/5], Loss: 0.0239
    Accuracy:  0.991

    ==============================================================
    ResNet34 Result
    ==============================================================
    Epoch [1/5], Loss: 0.4893
    Epoch [2/5], Loss: 0.0785
    Epoch [3/5], Loss: 0.0549
    Epoch [4/5], Loss: 0.0434
    Epoch [5/5], Loss: 0.0414
    Accuracy:  0.9841

    ==============================================================
    ResNet152 Result
    ==============================================================
    Epoch [1/5], Loss: 0.5145
    Epoch [2/5], Loss: 0.0766
    Epoch [3/5], Loss: 0.2529
    Epoch [4/5], Loss: 0.0849
    Epoch [5/5], Loss: 0.0465
    Accuracy:  0.982

    ==============================================================
    SqueezeNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1915
    Epoch [2/5], Loss: 0.0595
    Epoch [3/5], Loss: 0.0449
    Epoch [4/5], Loss: 0.0380
    Epoch [5/5], Loss: 0.0337
    Accuracy:  0.9889

	==============================================================
	ResNeXt Result
	==============================================================
	Epoch [1/5], Loss: 0.4990
	Epoch [2/5], Loss: 0.0730
	Epoch [3/5], Loss: 0.0501
	Epoch [4/5], Loss: 0.0412
	Epoch [5/5], Loss: 0.0426
	Accuracy:  0.9901

    ==============================================================
    DenseNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1579
    Epoch [2/5], Loss: 0.0538
    Epoch [3/5], Loss: 0.0423
    Epoch [4/5], Loss: 0.0385
    Epoch [5/5], Loss: 0.0336
    Accuracy:  0.9876

    ==============================================================
    WideResNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.2046
    Epoch [2/5], Loss: 0.0515
    Epoch [3/5], Loss: 0.0387
    Epoch [4/5], Loss: 0.0326
    Epoch [5/5], Loss: 0.0291
    Accuracy:  0.9922

    ==============================================================
    MobileNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.2715
    Epoch [2/5], Loss: 0.0862
    Epoch [3/5], Loss: 0.0683
    Epoch [4/5], Loss: 0.0600
    Epoch [5/5], Loss: 0.0534
    Accuracy:  0.9882

    ==============================================================
    NASNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1248
    Epoch [2/5], Loss: 0.0465
    Epoch [3/5], Loss: 0.0371
    Epoch [4/5], Loss: 0.0323
    Epoch [5/5], Loss: 0.0266
    Accuracy:  0.9869

    ==============================================================
    ShuffleNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.3571
    Epoch [2/5], Loss: 0.1175
    Epoch [3/5], Loss: 0.0955
    Epoch [4/5], Loss: 0.0818
    Epoch [5/5], Loss: 0.0737
    Accuracy:  0.974

    ==============================================================
    EfficientNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.2287
    Epoch [2/5], Loss: 0.0709
    Epoch [3/5], Loss: 0.0519
    Epoch [4/5], Loss: 0.0437
    Epoch [5/5], Loss: 0.0419
    Accuracy:  0.9843

    ==============================================================
    RegNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.3591
    Epoch [2/5], Loss: 0.0634
    Epoch [3/5], Loss: 0.0450
    Epoch [4/5], Loss: 0.0386
    Epoch [5/5], Loss: 0.2168
    Accuracy:  0.9666
    
    ==============================================================
    GhostNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1532
    Epoch [2/5], Loss: 0.0497
    Epoch [3/5], Loss: 0.0391
    Epoch [4/5], Loss: 0.0324
    Epoch [5/5], Loss: 0.0264
    Accuracy:  0.9873

    ==============================================================
    MicroNet Result
    ==============================================================
    Epoch [1/5], Loss: 0.1517
    Epoch [2/5], Loss: 0.0589
    Epoch [3/5], Loss: 0.0466
    Epoch [4/5], Loss: 0.0403
    Epoch [5/5], Loss: 0.0352
    Accuracy:  0.991
    """