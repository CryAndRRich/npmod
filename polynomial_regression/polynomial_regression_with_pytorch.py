import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

random.seed(777)
torch.manual_seed(777)

class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        #self.linear = nn.Linear(degree, 1)
        self.number_of_coefs = degree + 1
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float32))
            for _ in range(self.number_of_coefs)
        ])

    def forward(self, x):
        poly = 0
        for i in range(self.number_of_coefs):
            poly += self.coefficients[i] * (x ** i)
        #poly = self.linear(x)
        return poly
    
    def get_coefs(self):
        for i, coef in enumerate(self.coefficients):
            print('x{}={:.3f}'.format(i, coef.item()), end=' ')
            #x0 = self.linear.bias
            #x1,x2,... = self.linear.weight

def train_model_pytorch(x_train, y_train, learn_rate, number_of_epochs, degree=1):
    model = PolynomialRegressionModel(degree)
    #degree = 1 => LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum = 0.99)

    for epoch in range(number_of_epochs + 1):
        model.train()
        prediction = model(x_train)
        
        cost = func.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        model.eval()
        
        if epoch % 10000 == 0:
            print('Epoch: {:4d}/{} Cost: {:.6f}'.format(
                  epoch, number_of_epochs, cost.item()))
    
            model.get_coefs()
            print('')
