import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def loss_function(y, prob):
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

def gradient_descent(X, y, learn_rate, num_iterations):
    n_samples = X.shape[0]
    weight = np.zeros((X.shape[1] + 1, 1)) 
    y = y.reshape((-1, 1))
    for _ in range(num_iterations):
        x = np.hstack((np.ones((n_samples, 1)), X))  
        prob = sigmoid_function(np.dot(x, weight))
        gradient = np.dot(x.T, (prob - y)) / n_samples
        weight -= learn_rate * gradient

    return weight

def predict(X, weight):
    x = np.hstack((np.ones((X.shape[0], 1)), X))  
    prob = sigmoid_function(np.dot(x, weight))
    return (prob > 0.5).astype(int)

def draw_graph(X, y, weight):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X[y.flatten() == 0], np.zeros_like(X[y.flatten() == 0]), color='red', label='Class 0')
    plt.scatter(X[y.flatten() == 1], np.ones_like(X[y.flatten() == 1]), color='blue', label='Class 1')

    x_values = np.linspace(np.min(X) - 1, np.max(X) + 1, 100)
    y_values = sigmoid_function(weight[0] + weight[1] * x_values)
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    
def logistic_regression():
    points = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
    X = points[:, :-1]  
    y = points[:, -1] 
    learn_rate = 0.001
    num_iterations = 1000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    weight = gradient_descent(X_train, y_train, learn_rate, num_iterations) 

    y_pred = predict(X_test, weight)
    
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Hamming Loss: {hamming_loss(y_test, y_pred)}')

    draw_graph(X_test, y_test, weight)

if __name__ == '__main__':
    logistic_regression()
