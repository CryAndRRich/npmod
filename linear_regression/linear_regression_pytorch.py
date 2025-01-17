import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class LinearRegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)
    
class LinearRegressionPytorch(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        self.model = LinearRegressionModule()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        for _ in range(self.number_of_epochs):
            predictions = self.model(features)
            
            cost = func.mse_loss(predictions, labels)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        
        params = list(self.model.parameters())
        weight = params[0].item()
        bias = params[1].item()
    
        print("Epoch: {}/{} Weight: {:.5f}, Bias: {:.5f} Cost: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, weight, bias, cost))
    
    def __str__(self):
        return "Linear Regression (Pytorch)"