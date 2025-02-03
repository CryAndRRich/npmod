import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os

from modelML import *
from npmod import *

def test_npmod():
    training_epochs = 2
    batch_size = 100

    mnist_train = dsets.MNIST(root='MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    net = nn.Sequential(layers = [nn.Linear(28 * 28, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.Sigmoid(),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(64, 10)])

    criterion = nn.CE()
    optim = nn.SGD(net.get_layers(), learn_rate=0.001)

    for epoch in range(training_epochs):
        for features, labels in data_loader:
            # Reshape input image into [batch_size by 784]
            # Label is not one-hot encoded
            features = features.view(-1, 28 * 28).numpy()
            labels = labels.numpy()

            features = np.expand_dims(features, axis=1)
            labels = np.expand_dims(labels, axis=1)
            
            predictions = net(features)
            loss = criterion(predictions, labels)
            net.backward(criterion)
            optim.step()

    test_features = mnist_test.test_data.view(-1, 28 * 28).numpy()
    test_labels = mnist_test.test_labels.numpy()

    predictions = np.argmax(net(test_features), axis=1)
    accuracy = (predictions == test_labels).mean()
    print(accuracy)

def test_modelML():
    path = os.path.join('datasets', 'svm_data.csv')
    data = getData(data_path=path)

    features, test_features, labels, test_labels = data.get_processed_data()

    model = RandomForest()
    print(model)
    model.fit(features, labels)
    model.predict(test_features, test_labels)

test_modelML()
#Random Forest:
#- Decision Trees: C4.5 Algorithm
#- Decision Trees: ID3 Algorithm
#- Decision Trees: CHAID Algorithm
#- Decision Trees: CART Algorithm
#- Decision Trees: CART Algorithm
#- Decision Trees: C5.0/See5 Algorithm
#- Decision Trees: C4.5 Algorithm
#- Decision Trees: TAO Algorithm
#- Decision Trees: C4.5 Algorithm
#- Decision Trees: OC1 Algorithm

#Accuracy: 0.97500 F1-score: 0.97505