import torch
import torch.nn as nn
import torch.optim as optim
from base_model import ModelML

torch.manual_seed(42)

class LogisticRegressionModule(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(in_features=n, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

class LogisticRegressionPytorch(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        _, n = features.shape

        self.model = LogisticRegressionModule(n)
        self.criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learn_rate)

        self.model.train()
        for _ in range(self.number_of_epochs):
            prediction = self.model(features)
            
            cost = self.criterion(prediction, labels)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        self.model.eval()
    
    def predict(self, test_features, test_labels):
        with torch.no_grad():
            outputs = self.model(test_features)
            predictions = (outputs >= 0.5).float()
            predictions = predictions.detach().numpy()
            test_labels = test_labels.detach().numpy()

            accuracy, f1 = self.evaluate(predictions, test_labels)
            print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
                   self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Logistic Regression (Pytorch)"
    