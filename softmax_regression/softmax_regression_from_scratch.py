import numpy as np

np.random.seed(777)

def softmax_function(z):
    exp_z = np.exp(z - np.max(z)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_train, probs, number_of_samples):
    cost = - np.sum(y_train * np.log(probs)) / number_of_samples
    return cost

def one_hot_encode(y_train, number_of_samples, number_of_classes):
    one_hot = np.zeros((number_of_samples, number_of_classes))
    one_hot[np.arange(number_of_samples), y_train.T] = 1
    return one_hot

def gradient_descent(x_train, y_train, probs, weights, bias, learn_rate):
    m = x_train.shape[0]

    weights_gradient = np.dot(x_train.T, (probs - y_train)) / m
    bias_gradient = np.sum(probs - y_train, axis=0) / m
    
    weights -= learn_rate * weights_gradient.T
    bias -= learn_rate * bias_gradient

    return weights, bias

def train_model_scratch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs, number_of_classes=2):
    m, n = x_train.shape

    weights = np.random.rand(number_of_classes, n)
    bias = np.zeros((1, number_of_classes))

    for epoch in range(number_of_epochs + 1):
        y_one_hot = one_hot_encode(y_train, m, number_of_classes)

        scores = np.dot(x_train, weights.T) + bias
        probs = softmax_function(scores)
        cost = cross_entropy(y_one_hot, probs, m)

        weights, bias = gradient_descent(x_train, y_one_hot, probs, weights, bias, learn_rate)

        if epoch % 100 == 0:
            accuracy = test_model_scratch(x_test, y_test, weights, bias)
            print('Epoch: {:4d}/{} Cost: {:.6f} Accuracy: {:.2f}%'.format(
                epoch, number_of_epochs, cost, accuracy * 100))

def test_model_scratch(x_test, y_test, weights, bias):
    scores = np.dot(x_test, weights.T) + bias
    probs = softmax_function(scores)
    y_pred = np.argmax(probs, axis=1)[:, np.newaxis]
    accuracy = (np.sum(y_pred == y_test) / y_test.shape[0])
    return accuracy
