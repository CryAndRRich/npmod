import torch

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

    Returns:
        dist: The computed distance between the feature vectors and the test features
    """
    dist = (features - test_features).pow(p).sum(axis=1).pow(1 / p)
    return dist

def get_knn(features: torch.Tensor, 
            targets: torch.Tensor, 
            test_features: torch.Tensor, 
            k: int) -> torch.Tensor:
    """
    Performs K-Nearest Neighbors classification for a given test sample

    Parameters:
        features: The input data features
        targets: The targets corresponding to the input features
        test_features: The feature vector for the test sample
        k: The number of nearest neighbors to consider

    Returns:
        prediction: The predicted target for the test sample
    """
    test_features = test_features.unsqueeze(1).T
    dist = minkowski_dist(features, test_features)
    _, indices = torch.sort(dist)
    
    try:
        k_nearest = targets[indices][:k]
        prediction = torch.mode(k_nearest).values.item()
    except ValueError:
        raise ValueError("'{}' must be within range of 1 and {}".format(k, targets.shape[0]))

    return prediction

class KNearestNeighbors():
    def __init__(self, neighbors: int) -> None:
        """
        Initializes the K-Nearest Neighbors model

        Parameters:
            neighbors: The number of nearest neighbors to consider
        """
        self.k = neighbors
    
    def fit(self, 
            features: torch.Tensor, 
            targets: torch.Tensor) -> None:
        """
        Fits the K-Nearest Neighbors model to the input data

        Parameters:
            features: Feature matrix of the training data
            targets: Array of targets corresponding to the training data
        """
        self.features = features
        self.targets = targets
    
    def predict(self, test_features: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions on the test set and evaluates the model

        Parameters:
            test_features: The input features for testing

        Returns:
            predictions: The prediction targets
        """
        predictions = torch.zeros(test_features.shape[0])

        for i in range(test_features.shape[0]):
            predictions[i] = get_knn(self.features, self.targets, test_features[i], self.k)

        predictions = predictions.detach().numpy()
    
        return predictions
    
    def __str__(self) -> str:
        return "K Nearest Neighbors"
