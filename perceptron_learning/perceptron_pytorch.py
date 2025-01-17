import torch
import torch.nn as nn
from base_model import ModelML

torch.manual_seed(42)

class PerceptionModule(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(in_features=n, out_features=1)

    def heaviside_step(self, weighted_sum):
        weighted_sum = [int(weight >= 0) for weight in weighted_sum]
        return torch.tensor(weighted_sum)

    def forward(self, x):
        weighted_sum = self.linear(x)
        return self.heaviside_step(weighted_sum)

class PerceptronLearningPytorch(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        _, n = features.shape
        self.model = PerceptionModule(n)

        self.model.train()
        for _ in range(self.number_of_epochs):
            cost = 0
            for x, y in zip(features, labels):
                predictions = self.model(x)
                error = predictions - y
                cost += error

                weights = self.model.linear.weight
                bias = self.model.linear.bias

                weights = weights - self.learn_rate * error * x
                bias = bias - self.learn_rate * error

                self.model.linear.weight = nn.Parameter(weights)
                self.model.linear.bias = nn.Parameter(bias)
        self.model.eval()
    
    def predict(self, test_features, test_labels):
        with torch.no_grad():
            predictions = self.model(test_features)
            predictions = predictions.detach().numpy()
            test_labels = test_labels.detach().numpy()

            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Perceptron Learning Algorithm (Pytorch)"
    