import torch
import torch.nn as nn

torch.manual_seed(777)

class PerceptionModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, 1)

    def heaviside_step(self, weighted_sum):
        weighted_sum = [int(weight >= 0) for weight in weighted_sum]
        return torch.tensor(weighted_sum)

    def forward(self, x):
        weighted_sum = self.linear(x)
        return self.heaviside_step(weighted_sum)

def train_model_pytorch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs):
    _, n = x_train.shape
    model = PerceptionModel(n)

    for epoch in range(number_of_epochs + 1):
        cost = 0
        for x, y in zip(x_train, y_train):
            prediction = model(x)
            error = prediction - y
            cost += error

            weights = model.linear.weight
            bias = model.linear.bias

            weights = weights - learn_rate * error * x
            bias = bias - learn_rate * error

            model.linear.weight = nn.Parameter(weights)
            model.linear.bias = nn.Parameter(bias)
        
        accuracy = test_model_pytorch(model, x_test, y_test)
        print('Epoch: {:2d}/{} Weight: {} Bias: {} Cost: {:.6f} Accuracy: {:.2f}%'.format(
               epoch, number_of_epochs, weights.detach().numpy(), bias.detach().numpy(),cost.item(), accuracy))

def test_model_pytorch(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        accuracy = (y_pred == y_test[:,0]).float().mean()
    
    return accuracy * 100
