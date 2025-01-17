import torch
from base_model import ModelML

def minkowski_dist(features, test_features, p=2):
    dist = (features - test_features).pow(p).sum(axis=1).pow(1 / p)
    return dist

def get_knn(features, labels, test_features, k):
    test_features = test_features.unsqueeze(1).T
    dist = minkowski_dist(features, test_features)
    _, indices = torch.sort(dist)
    
    try:
        k_nearest = labels[indices][:k]
        prediction = k_nearest.sum() >= (k // 2)
    except ValueError:
        raise ValueError("'{}' must be within range of 1 and {}".format(k, labels.shape[0]))

    return prediction

class KNearestNeighbors(ModelML):
    def __init__(self, neighbors):
        self.k = neighbors
    
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
    
    def predict(self, test_features, test_labels):
        predictions = torch.zeros(test_labels.shape[0])

        for i in range(test_features.shape[0]):
            predictions[i] = get_knn(self.features, self.labels, test_features[i], self.k)

        predictions = predictions.detach().numpy()
        test_labels = test_labels.detach().numpy()

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("k: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.k, accuracy, f1))
    
    def __str__(self):
        return "K Nearest Neighbors"

