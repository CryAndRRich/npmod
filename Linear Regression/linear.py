from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def cost_function(b, m, points):
    total_error = 0

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - (m * x + b)) ** 2
    
    return total_error / float(len(points))

def step_gradient(b, m, points, learn_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]

        b_gradient += -(2 / n) * (y - ((m * x) + b))
        m_gradient += -(2 / n) * x * (y - ((m * x) + b))
    
    b -= (learn_rate * b_gradient)
    m -= (learn_rate * m_gradient)

    return [b, m]

def gradient_descent(points, b, m, learn_rate, num):
    for _ in range(num):
        b , m = step_gradient(b, m, array(points), learn_rate)
    
    return [b,m]

def draw_graph(points, b, m):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Data Points')
    
    x_values = np.array([min(points[:, 0]), max(points[:, 0])])
    y_values = m * x_values + b
    plt.plot(x_values, y_values, color='red', label='Regression Line')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def linear_regression():
    points = np.genfromtxt('data.csv', delimiter = ',')
    learn_rate = 0.0001
    b_random, m_random = 0, 0  # y = mx + b
    num_iterations = 1000

    #train model
    print(f'b = {b_random}, m = {m_random}, error = {cost_function(b_random, m_random, points)}')
    [b, m] = gradient_descent(points, b_random, m_random, learn_rate, num_iterations)
    print('Running...')
    print(f'After {num_iterations} iterations we have:')
    print(f'b = {b}, m = {m}, error = {cost_function(b, m, points)}')

    #draw_graph(points, b, m)

if __name__ == '__main__':
    linear_regression()