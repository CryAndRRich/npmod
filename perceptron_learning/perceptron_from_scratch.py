import numpy as np

np.random.seed(777)

def cost_function(y_train, y_predict):
    cost = y_predict - y_train
    return cost

def heaviside_step(x_train, weights, bias):
    weighted_sum = x_train @ weights.T + bias
    try:
        return [int(weight >= 0) for weight in weighted_sum]
    except:
        return int(weighted_sum >= 0)


def train_model_scratch(x_train, y_train, x_test, y_test, learn_rate, number_of_epochs):
    _, n = x_train.shape

    weights, bias = np.random.rand(n), 0

    for epoch in range(number_of_epochs + 1):
        cost = 0
        for x, y in zip(x_train, y_train):
            prediction = heaviside_step(x, weights, bias)
            error = cost_function(y, prediction)
            cost += error

            weights += learn_rate * error * x
            bias += learn_rate * error

        accuracy = test_model_scratch(x_test, y_test, weights, bias)
        print('Epoch: {:2d}/{} Weight: [{}] Bias: [{:.4f}] Cost: {:.6f} Accuracy: {:.2f}%'.format(
               epoch, number_of_epochs, weights, bias, cost, accuracy))
        
def test_model_scratch(x_test, y_test, weights, bias):
    y_predict = heaviside_step(x_test, weights, bias)
    accuracy = ((y_predict != y_test).sum() / y_test.shape[0]).item()
    return accuracy * 100

