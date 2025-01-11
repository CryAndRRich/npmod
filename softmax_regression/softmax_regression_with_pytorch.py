import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

random.seed(777)
torch.manual_seed(777)

class SoftmaxRegressionModel(nn.Module):
    def __init__(self, number_of_features, number_of_classes):
        super().__init__()
        self.linear = nn.Linear(number_of_features, number_of_classes)

    def forward(self, x):
        return self.linear(x)
    
def train_model_pytorch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs, number_of_classes=2):
    _, n = x_train.shape
    model = SoftmaxRegressionModel(n, number_of_classes)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    for epoch in range(number_of_epochs + 1):
        prediction = model(x_train)
        
        cost = func.cross_entropy(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            accuracy = test_model_pytorch(model, x_test, y_test)
            print('Epoch: {:4d}/{} Cost: {:.6f} Accuracy: {:.2f}%'.format(
                  epoch, number_of_epochs, cost.item(), accuracy * 100))

def test_model_pytorch(model, x_test, y_test):
    prediction = model(x_test)
    y_pred = prediction.max(1)[1]
    accuracy = ((y_pred == y_test).sum() / y_test.shape[0]).item()

    return accuracy