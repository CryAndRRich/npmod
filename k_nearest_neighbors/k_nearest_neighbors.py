import torch

class KNearestNeighbors():
    def __init__(self, neighbors):
        self.k = neighbors
    
    def minkowski_dist(self, x_train, x_test, p=2):
        dist = (x_train - x_test).pow(p).sum(axis=1).pow(1 / p)
        return dist
    
    def get_knn(self, x_train, y_train, x_test):
        x_test = x_test.unsqueeze(1).T
        dist = self.minkowski_dist(x_train, x_test)
        _, indices = torch.sort(dist)
        
        k_nearest = y_train[indices][:self.k]
        prediction = k_nearest.sum() >= (self.k // 2)

        return prediction
    
def train_model(x_train, y_train, x_test, y_test, k_neighbors):
    for k in range(0, k_neighbors + 1, 5):
        knn = KNearestNeighbors(k)
        y_predict = torch.zeros(y_test.shape[0])

        for i in range(x_test.shape[0]):
            y_predict[i] = knn.get_knn(x_train, y_train, x_test[i])
        
        accuracy = test_model(y_test, y_predict) * 100
        print('KNN: {} Accuracy: {}%'.format(k, accuracy))
    
def test_model(y_test, y_predict):
    accuracy = 0
    for i in range(y_test.shape[0]):
        accuracy += int(y_test[i] == y_predict[i])
    accuracy /= (y_test.shape[0])

    return accuracy