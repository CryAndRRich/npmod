import numpy as np

def minkowski_dist(features: np.ndarray,
                   test_features: np.ndarray,
                   p: int = 2) -> np.ndarray:
    """
    Computes the Minkowski distance between the feature vectors and the test features.

    Parameters:
        features: Training feature matrix (N, D)
        test_features: Single test feature vector (D,)
        p: Order of Minkowski distance (default is 2 = Euclidean)

    Returns:
        dist: Array of distances (N,)
    """
    dist = np.sum(np.abs(features - test_features) ** p, axis=1) ** (1 / p)
    return dist

def get_knn(features: np.ndarray,
            targets: np.ndarray,
            test_features: np.ndarray,
            k: int,
            weights: str = "uniform") -> int:
    """
    Performs KNN classification for a single test point.

    Parameters:
        features: Training feature matrix
        targets: Training labels
        test_features: Single test sample
        k: Number of nearest neighbors
        weights: Weighting scheme for neighbors

    Returns:
        prediction: Predicted label for the test sample
    """
    dist = minkowski_dist(features, test_features)
    indices = np.argsort(dist)
    k_nearest_targets = targets[indices[:k]]
    k_nearest_dists = dist[indices[:k]]

    if weights == "uniform":
        prediction = np.mean(k_nearest_targets)
    elif weights == "distance":
        weights = 1 / (k_nearest_dists + 1e-5)
        weighted_sum = np.sum(k_nearest_targets * weights)
        prediction = weighted_sum / np.sum(weights)
    else:
        raise ValueError("weights must be 'uniform' or 'distance'")

    return prediction

class KNeighborsRegressor:
    def __init__(self, 
                 neighbors: int,
                 weights: str = "uniform") -> None:
        """
        Initializes the KNN model.

        Parameters:
            neighbors: Number of nearest neighbors (k)
            weights: Weighting scheme for neighbors
        """
        if weights not in ["uniform", "distance"]:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = neighbors
        self.weights = weights

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Stores the training data.

        Parameters:
            features: Training feature matrix
            targets: Training labels
        """
        self.features = features
        self.targets = targets

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for a batch of test samples.

        Parameters:
            test_features: Test feature matrix (M, D)

        Returns:
            predictions: Predicted labels (M,)
        """
        predictions = np.zeros(test_features.shape[0])
        for i in range(test_features.shape[0]):
            predictions[i] = get_knn(self.features, self.targets, test_features[i], self.k, self.weights)
        return predictions

    def __str__(self) -> str:
        return "K Nearest Neighbors Regressor"
