def cost_function(x_train, y_train, weight, bias, batch_size):
    total_cost = 0

    for i in range(batch_size):
        x = x_train[i]
        y = y_train[i]
        total_cost += (y - (weight * x + bias)) ** 2
    
    avg_cost = total_cost / batch_size
    return avg_cost

def gradient_descent(x_train, y_train, weight, bias, learn_rate, batch_size):
    weight_gradient = 0
    bias_gradient = 0

    for i in range(batch_size):
        x = x_train[i]
        y = y_train[i]

        weight_gradient += -(2 / batch_size) * x * (y - ((weight * x) + bias))
        bias_gradient += -(2 / batch_size) * (y - ((weight * x) + bias))
    
    weight -= (learn_rate * weight_gradient)
    bias -= (learn_rate * bias_gradient)

    return weight, bias

def train_model_scratch(x_train, y_train, learn_rate, batch_size, number_of_epochs):
    weight, bias = 0, 0
    for epoch in range(number_of_epochs + 1):
        weight, bias = gradient_descent(x_train, y_train, weight, bias, learn_rate, batch_size)

        if epoch % 100 == 0:
            cost = cost_function(x_train, y_train, weight, bias, batch_size)
            print('Epoch: {:4d}/{} Weight: {:.3f}, Bias: {:.3f} Cost: {:.6f}'.format(
                  epoch, number_of_epochs, weight, bias, cost))
            
