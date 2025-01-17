import numpy as np
from base_model import ModelML

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def log_loss(x, y):
    return -(x * np.log(y)) - ((1 - x) * np.log(1 - y))

def cost_function(features, labels, weight, bias):
    m, n = features.shape
    prob = np.zeros(m)
    for i in range(m):
        predict = 0
        for j in range(n):
            predict += features[i, j] * weight[j]
        prob[i] = sigmoid_function(predict + bias)
    
    cost = np.mean(log_loss(labels, prob))
    return cost 

def gradient_descent(features, labels, weight, bias, learn_rate):
    m, n = features.shape

    weight_gradient = np.zeros(n)
    bias_gradient = 0
    for i in range(m):
        predict = 0
        for j in range(n):
            predict += features[i, j] * weight[j]
        prob = sigmoid_function(predict + bias)

        for j in range(n):
            weight_gradient[j] += (prob - labels[i]) * features[i, j]
        bias_gradient += prob - labels[i]

    weight_gradient /= m
    bias_gradient /= m

    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

class LogisticRegressionNumpy(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        _, n = features.shape

        self.weight = np.zeros(n)
        self.bias = 0

        for _ in range(self.number_of_epochs):
            cost = cost_function(features, labels, self.weight, self.bias)
            self.weight, self.bias = gradient_descent(features, labels, self.weight, self.bias, self.learn_rate)

    def predict(self, test_features, test_labels):
        prob = sigmoid_function(np.dot(test_features, self.weight)) + self.bias
        predictions = (prob >= 0.5).astype(int)

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Logistic Regression (Numpy)"
    