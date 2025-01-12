import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

class SVMModel(nn.Module):
    def __init__(self, x_train, kernel='rbf', gamma=1.0, train_gamma=True, degree=4, b=1.0):
        super().__init__()
        self.train_data = x_train
        self.gamma = torch.nn.Parameter(torch.FloatTensor([gamma]), requires_grad=train_gamma)
        self.b = b

        if kernel == 'linear':
            self.kernel = self.linear
            self.C = x_train.size(1)
        elif kernel == 'rbf':
            self.kernel = self.rbf
            self.C = x_train.size(0)
        elif kernel == 'poly':
            self.kernel = self.poly
            self.C = x_train.size(0)
            self.degree = degree
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid
            self.C = x_train.size(0)
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

    
def train_model(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs):
    model_linear = SVMModel(x_train, kernel='linear')
    optimizer_linear = optim.SGD(model_linear.parameters(), lr=learn_rate)
    
    model_rbf= SVMModel(x_train, kernel='rbf')
    optimizer_rbf = optim.SGD(model_rbf.parameters(), lr=learn_rate)
    
    model_poly = SVMModel(x_train, kernel='poly')
    optimizer_poly = optim.SGD(model_poly.parameters(), lr=learn_rate)
    
    model_sigmoid= SVMModel(x_train, kernel='sigmoid')
    optimizer_sigmoid = optim.SGD(model_sigmoid.parameters(), lr=learn_rate)

    for epoch in range(number_of_epochs + 1):
        linear_prediction = model_linear(x_train)
        rbf_prediction = model_rbf(x_train)
        poly_prediction = model_poly(x_train)
        sigmoid_prediction = model_sigmoid(x_train)
        
        linear_cost = model_linear.hinge_loss(linear_prediction, y_train.unsqueeze(1))
        rbf_cost = model_rbf.hinge_loss(rbf_prediction, y_train.unsqueeze(1))
        poly_cost = model_poly.hinge_loss(poly_prediction, y_train.unsqueeze(1))
        sigmoid_cost = model_sigmoid.hinge_loss(sigmoid_prediction, y_train.unsqueeze(1))
        
        optimizer_linear.zero_grad()
        optimizer_rbf.zero_grad()
        optimizer_poly.zero_grad()
        optimizer_sigmoid.zero_grad()

        linear_cost.backward()
        rbf_cost.backward()
        poly_cost.backward()
        sigmoid_cost.backward()

        optimizer_linear.step()
        optimizer_rbf.step()
        optimizer_poly.step()
        optimizer_sigmoid.step()
        
        if epoch % 100 == 0:
            print('Epoch: {:4d}/{} Linear Cost: {:.4f} RBF Cost: {:.4f} Polynomial Cost: {:.4f} Sigmoid Cost: {:.4f}'.format(
                  epoch, number_of_epochs, linear_cost.item(), rbf_cost.item(), poly_cost.item(), sigmoid_cost.item()))
            