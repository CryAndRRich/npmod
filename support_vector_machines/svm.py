import torch
import torch.nn as nn
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class SVMModule(nn.Module):
    def __init__(self, features, kernel="rbf", gamma=0.1, train_gamma=True, degree=3, b=1.0):
        super().__init__()
        self.train_data = features
        self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]), requires_grad=train_gamma)
        self.b = b

        if kernel == "linear":
            self.kernel = self.linear
            self.C = features.size(1)
        elif kernel == "rbf":
            self.kernel = self.rbf
            self.C = features.size(0)
        elif kernel == "poly":
            self.kernel = self.poly
            self.C = features.size(0)
            self.degree = degree
        elif kernel == "sigmoid":
            self.kernel = self.sigmoid
            self.C = features.size(0)
        else: 
            raise ValueError(f"Unsupported kernel '{kernel}'. Supported kernels are 'linear', 'rbf', 'poly', and 'sigmoid'")

        self.weight = nn.Linear(in_features=self.C, out_features=1)

    def rbf(self, x):
        y = self.train_data.repeat(x.size(0), 1, 1)
        return torch.exp(-self.gamma * ((x[:, None] - y) ** 2).sum(dim=2))
    
    def linear(self, x):
        return x
    
    def poly(self, x):
        y = self.train_data.repeat(x.size(0), 1, 1)
        return (self.gamma * torch.bmm(x.unsqueeze(1), y.transpose(1, 2)) + self.b).pow(self.degree).squeeze(1)
    
    def sigmoid(self, x):
        y = self.train_data.repeat(x.size(0), 1, 1)
        return torch.tanh(self.gamma * torch.bmm(x.unsqueeze(1), y.transpose(1, 2)) + self.b).squeeze(1)
    
    def hinge_loss(self, x, y):
        return torch.max(torch.zeros_like(y), 1 - y * x).mean()
    
    def forward(self, x):
        y = self.kernel(x)
        y = self.weight(y)
        return y

class SVMModel(ModelML):
    def __init__(self, learn_rate, number_of_epochs, kernel="linear", gamma=0.1):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, features, labels):
        self.model = SVMModule(features, self.kernel, self.gamma)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.model.train()
        for _ in range(self.number_of_epochs):
            predictions = self.model(features)
            cost = self.model.hinge_loss(predictions, labels.unsqueeze(1))

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        self.model.eval()
    
    def predict(self, test_features, test_labels):
        with torch.no_grad():
            predictions = (self.model(test_features)).detach().numpy()
            predictions = (predictions > predictions.mean())
            test_labels = test_labels.detach().numpy()
            
            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Support Vector Machine (SVM) - Kernel:'{}'".format(self.kernel)