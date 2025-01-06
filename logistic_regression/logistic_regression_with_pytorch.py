import random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(777)
torch.manual_seed(777)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        prediction = self.sigmoid(self.linear(x))
        return prediction

def train_model_pytorch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs):
    model = LogisticRegressionModel()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    for epoch in range(number_of_epochs + 1):
        prediction = model(x_train)
        
        cost = criterion(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            accuracy = test_model_pytorch(model, x_test, y_test)
            print('Epoch: {:4d}/{} Cost: {:.6f} Accuracy: {:.2f}%'.format(
                  epoch, number_of_epochs, cost.item(), accuracy))

def test_model_pytorch(model, x_test, y_test):
    with torch.no_grad():
        outputs = model(x_test)
        predictions = (outputs >= 0.5).float()
        accuracy = ((predictions == y_test).sum() / y_test.shape[0]).item()
    
    return accuracy * 100
