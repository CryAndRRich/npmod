import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def log_loss(x, y):
    return -(x * np.log(y)) - ((1 - x) * np.log(1 - y))

def cost_function(x_train, y_train, weight, bias):
    m, n = x_train.shape
    prob = np.zeros(m)
    for i in range(m):
        predict = 0
        for j in range(n):
            predict += x_train[i, j] * weight[j]
        prob[i] = sigmoid_function(predict + bias)
    
    cost = np.mean(log_loss(y_train, prob))
    return cost 

def gradient_descent(x_train, y_train, weight, bias, learn_rate):
    m, n = x_train.shape

    weight_gradient = np.zeros(n)
    bias_gradient = 0
    for i in range(m):
        predict = 0
        for j in range(n):
            predict += x_train[i, j] * weight[j]
        prob = sigmoid_function(predict + bias)

        for j in range(n):
            weight_gradient[j] += (prob - y_train[i]) * x_train[i, j]
        bias_gradient += prob - y_train[i]

    weight_gradient /= m
    bias_gradient /= m

    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

def train_model_scratch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs):
    _, n = x_train.shape
    weight = np.zeros(n)
    bias = 0

    for epoch in range(number_of_epochs + 1):
        weight, bias = gradient_descent(x_train, y_train, weight, bias, learn_rate)

        if epoch % 100 == 0:
            cost = cost_function(x_train, y_train, weight, bias)
            prob = sigmoid_function(np.dot(x_test, weight)) + bias
            prediction = (prob >= 0.5).astype(int)
            accuracy = test_model_scratch(y_test, prediction)
            print('Epoch: {:4d}/{} Cost: {:.6f} Accuracy: {:.2f}%'.format(
                  epoch, number_of_epochs, cost, accuracy))

def test_model_scratch(y_test, prediction):
    accuracy = ((prediction == y_test).sum() / y_test.shape[0]).item()
    return accuracy * 100
