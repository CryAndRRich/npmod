import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

random.seed(777)
torch.manual_seed(777)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_model_pytorch(x_train, y_train, number_of_epochs, learn_rate):
    model=LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    for epoch in range(number_of_epochs + 1):
        
        prediction = model(x_train)
        
        cost = func.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            params = list(model.parameters())
            m = params[0].item()
            b = params[1].item()
            print('Epoch: {:4d}/{} m: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                  epoch, number_of_epochs, m, b, cost.item()))