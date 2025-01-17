import numpy as np
from base_model import ModelML

np.random.seed(42)

def softmax_function(z):
    exp_z = np.exp(z - np.max(z)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(labels, probs, number_of_samples):
    cost = - np.sum(labels * np.log(probs)) / number_of_samples
    return cost

def one_hot_encode(labels, number_of_samples, number_of_classes):
    one_hot = np.zeros((number_of_samples, number_of_classes))
    one_hot[np.arange(number_of_samples), labels.T] = 1
    return one_hot

def gradient_descent(features, labels, probs, weights, bias, learn_rate):
    m = features.shape[0]

    weights_gradient = np.dot(features.T, (probs - labels)) / m
    bias_gradient = np.sum(probs - labels, axis=0) / m
    
    weights -= learn_rate * weights_gradient.T
    bias -= learn_rate * bias_gradient

    return weights, bias

class SoftmaxRegressionNumpy(ModelML):
    def __init__(self, learn_rate, number_of_epochs, number_of_classes=2):
        self.learn_rate = learn_rate
        self.number_of_epochs = number_of_epochs
        self.number_of_classes = number_of_classes
    
    def fit(self, features, labels):
        m, n = features.shape

        self.weights = np.random.rand(self.number_of_classes, n)
        self.bias = np.zeros((1, self.number_of_classes))

        for _ in range(self.number_of_epochs):
            y_one_hot = one_hot_encode(labels, m, self.number_of_classes)

            scores = np.dot(features, self.weights.T) + self.bias
            probs = softmax_function(scores)
            cost = cross_entropy(y_one_hot, probs, m)

            self.weights, self.bias = gradient_descent(features, y_one_hot, probs, self.weights, self.bias, self.learn_rate)
    
    def predict(self, test_features, test_labels):
        scores = np.dot(test_features, self.weights.T) + self.bias
        probs = softmax_function(scores)
        predictions = np.argmax(probs, axis=1)[:, np.newaxis]

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("Epoch: {}/{} Accuracy: {:.5f} F1-score: {:.5f}".format(
               self.number_of_epochs, self.number_of_epochs, accuracy, f1))
    
    def __str__(self):
        return "Softmax Regression (Numpy)"