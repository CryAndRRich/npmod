def cost_function(x_train, y_train, m, b, batch_size):
    total_cost = 0

    for i in range(batch_size):
        x = x_train[i]
        y = y_train[i]
        total_cost += (y - (m * x + b)) ** 2
    
    avg_cost = total_cost / batch_size
    return avg_cost

def step_gradient(x_train, y_train, m, b, learn_rate, batch_size):
    m_gradient = 0
    b_gradient = 0

    for i in range(batch_size):
        x = x_train[i]
        y = y_train[i]

        m_gradient += -(2 / batch_size) * x * (y - ((m * x) + b))
        b_gradient += -(2 / batch_size) * (y - ((m * x) + b))
    
    m -= (learn_rate * m_gradient)
    b -= (learn_rate * b_gradient)

    return m, b

def train_model_scratch(x_train, y_train, learn_rate, number_of_epochs, batch_size):
    m, b = 0, 0
    for epoch in range(number_of_epochs + 1):
        m, b = step_gradient(x_train, y_train, m, b, learn_rate, batch_size)

        if epoch % 100 == 0:
            cost = cost_function(x_train, y_train, m, b, batch_size)
            print('Epoch: {:4d}/{} m: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                  epoch, number_of_epochs, m, b, cost))



