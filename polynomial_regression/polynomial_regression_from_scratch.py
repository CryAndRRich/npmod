import numpy as np

def get_feature(x, degree=1):
    x_features = []
    for i in range(1, degree + 1):
        x_features.append(x ** i)
    
    x_features = np.array(x_features)
    return x_features

def cost_function(x_train, y_train, weight, bias, batch_size, degree=1):
    total_cost = 0

    for i in range(batch_size):
        x = get_feature(x_train[i], degree)
        y = y_train[i]
        total_cost += (y - (np.dot(x.T, weight) + bias)) ** 2
    
    avg_cost = total_cost / batch_size
    return avg_cost

def gradient_descent(x_train, y_train, weight, bias, learn_rate, batch_size, degree=1):
    weight_gradient = np.zeros(degree)
    bias_gradient = 0

    for i in range(batch_size):
        x = get_feature(x_train[i], degree)
        y = y_train[i]

        predict = np.dot(x.T, weight) + bias
        for j in range(degree):
            weight_gradient[j] += (predict - y) * x[j]
        bias_gradient += predict - y

    weight_gradient /= batch_size
    bias_gradient /= batch_size
    
    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

def train_model_scratch(x_train, y_train, learn_rate, batch_size, number_of_epochs, degree=1):
    weight, bias = np.zeros(degree), 0
    for epoch in range(number_of_epochs + 1):
        weight, bias = gradient_descent(x_train, y_train, weight, bias, learn_rate, batch_size, degree)

        if epoch % 10000 == 0:
            cost = cost_function(x_train, y_train, weight, bias, batch_size, degree)
            print('Epoch: {:4d}/{} Cost: {:.6f}'.format(
                  epoch, number_of_epochs, cost))
    
    print('x0={:.3f}'.format(bias), end=' ')
    for i in range(degree):
        print('x{}:{:.3f}'.format(i + 1, weight[i]), end=' ')



