import numpy as np
from base_model import ModelML

np.random.seed(42)

def cost_function(predictions, labels):
    cost = predictions - labels
    return cost

def heaviside_step(x_train, weights, bias):
    weighted_sum = x_train @ weights.T + bias
    try:
        return [int(weight >= 0) for weight in weighted_sum]
    except:
        return int(weighted_sum >= 0)

class PerceptronLearningNumpy(ModelML):
    def __init__(self, learn_rate, number_of_epochs):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
    
    def fit(self, features, labels):
        _, n = features.shape

        self.weights = np.random.rand(n)
        self.bias = 0

        for _ in range(self.number_of_epochs):
            cost = 0
            for x, y in zip(features, labels):
                predictions = heaviside_step(x, self.weights, self.bias)
                error = cost_function(predictions, y)
                cost += error

                self.weights += self.learn_rate * error * x
                self.bias += self.learn_rate * error
            
    def predict(self, test_features, test_labels):
        predictions = 1 - np.array(heaviside_step(test_features, self.weights, self.bias))

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Perceptron Learning Algorithm (Numpy)"