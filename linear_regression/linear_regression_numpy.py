from base_model import ModelML

def cost_function(features, labels, weight, bias):
    m = features.shape[0]
    total_cost = 0

    for i in range(m):
        x = features[i]
        y = labels[i]
        total_cost += (y - (weight * x + bias)) ** 2
    
    avg_cost = total_cost / m
    return avg_cost

def gradient_descent(features, labels, weight, bias, learn_rate):
    m = features.shape[0]

    weight_gradient = 0
    bias_gradient = 0

    for i in range(m):
        x = features[i]
        y = labels[i]

        weight_gradient += -(2 / m) * x * (y - ((weight * x) + bias))
        bias_gradient += -(2 / m) * (y - ((weight * x) + bias))
    
    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

class LinearRegressionNumpy(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        self.weight = 0
        self.bias = 0

        for _ in range(self.number_of_epochs):
            self.cost = cost_function(features, labels, self.weight, self.bias)
            self.weight, self.bias = gradient_descent(features, labels, self.weight, self.bias, self.learn_rate)
        
        print("Epoch: {}/{} Weight: {:.5f}, Bias: {:.5f} Cost: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, self.weight, self.bias, self.cost))
    
    def __str__(self):
        return "Linear Regression (Numpy)"