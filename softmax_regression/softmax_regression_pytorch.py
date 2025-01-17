import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class SoftmaxRegressionModule(nn.Module):
    def __init__(self, number_of_features, number_of_classes):
        super().__init__()
        self.linear = nn.Linear(in_features=number_of_features, out_features=number_of_classes)

    def forward(self, x):
        return self.linear(x)

class SoftmaxRegressionPytorch(ModelML):
    def __init__(self, learn_rate, number_of_epochs, number_of_classes=2):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.number_of_classes = number_of_classes
    
    def fit(self, features, labels):
        _, n = features.shape
        self.model = SoftmaxRegressionModule(n, self.number_of_classes)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.model.train()
        for _ in range(self.number_of_epochs):
            predictions = self.model(features)
            
            cost = func.cross_entropy(predictions, labels)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        self.model.eval()
    
    def predict(self, test_features, test_labels):
        with torch.no_grad():
            predictions = self.model(test_features).max(1)[1]
            predictions = predictions.detach().numpy()
            test_labels = test_labels.detach().numpy()

            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Softmax Regression (Pytorch)"