import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(11)
means = [[2, 2], [8, 3], [3, 6], [1, 9]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)

X = np.concatenate((X0, X1, X2, X3), axis = 0)
K = 4
original_label = np.asarray([0]*N + [1]*N + [2]*N + [3]*N).T

def graph(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    X3 = X[label == 3, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)
    plt.plot(X3[:, 0], X3[:, 1], 'm*', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()

def assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis = 1)

def update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def k_means_clustering():
    global X, K, original_label
    centers = [X[np.random.choice(X.shape[0], K, replace=False)]]
    labels = []
    iterations = 0 
    while True:
        labels.append(assign_labels(X, centers[-1]))
        new_centers = update_centers(X, labels[-1], K)
        if converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        iterations += 1
    
    print(f'Centers found by algorithm after {iterations} iterations:')
    print(centers[-1])

    #graph(X, original_label)

if __name__ == '__main__':
    k_means_clustering()