import torch
from base_model import ModelML

def minkowski_dist(features: torch.Tensor, 
                   test_features: torch.Tensor, 
                   p: int = 2) -> torch.Tensor:
    """
    Computes the Minkowski distance between the feature vectors and the test features

    Parameters:
    features: The input data features
    test_features: The feature vector for the test sample
    p: The power parameter for the Minkowski distance 
    (default is 2, which corresponds to the Euclidean distance)

    --------------------------------------------------
    Returns:
    dist: The computed distance between the feature vectors and the test features
    """
    dist = (features - test_features).pow(p).sum(axis=1).pow(1 / p)
    return dist

def get_knn(features: torch.Tensor, 
            labels: torch.Tensor, 
            test_features: torch.Tensor, 
            k: int) -> torch.Tensor:
    """
    Performs K-Nearest Neighbors classification for a given test sample

    Parameters:
    features: The input data features
    labels: The labels corresponding to the input features
    test_features: The feature vector for the test sample
    k: The number of nearest neighbors to consider

    --------------------------------------------------
    Returns:
    prediction: The predicted label for the test sample
    """
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
    def __init__(self, neighbors: int):
        """
        Initializes the K-Nearest Neighbors model

        Parameters:
        neighbors: The number of nearest neighbors to consider
        """
        self.k = neighbors
    
    def fit(self, 
            features: torch.Tensor, 
            labels: torch.Tensor) -> None:
        """
        Fits the K-Nearest Neighbors model to the input data

        Parameters:
        features: Feature matrix of the training data
        labels: Array of labels corresponding to the training data
        """
        self.features = features
        self.labels = labels
    
    def predict(self, 
                test_features: torch.Tensor, 
                test_labels: torch.Tensor) -> None:
        """
        Predicts the labels for the test data using the trained K-Nearest Neighbors model

        Parameters:
        test_features: Feature matrix of the test data
        test_labels: Array of labels corresponding to the test data
        """
        predictions = torch.zeros(test_labels.shape[0])

        for i in range(test_features.shape[0]):
            predictions[i] = get_knn(self.features, self.labels, test_features[i], self.k)

        predictions = predictions.detach().numpy()
        test_labels = test_labels.detach().numpy()

        accuracy, f1 = self.evaluate(predictions, test_labels)
        print("k: {} Accuracy: {:.5f} F1-score: {:.5f}".format(self.k, accuracy, f1))
    
    def __str__(self) -> str:
        return "K Nearest Neighbors"
