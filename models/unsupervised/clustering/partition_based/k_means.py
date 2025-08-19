from typing import Tuple
import numpy as np

def expectation_step(features: np.ndarray,
                     centroids: np.ndarray,
                     dists: np.ndarray,
                     number_of_clusters: int,
                     weights: np.ndarray = None,
                     beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the distance from each feature to each centroid and 
    assigns each feature to the closest centroid

    Parameters:
        features: The input data features
        centroids: The current centroids of the clusters
        dists: Preallocated array to store distances from each sample to each centroid
        number_of_clusters: The number of clusters
        weights: Feature weights
        beta: Weight exponent for WK-means

    Returns:
        dists_min: The minimum distance of each sample to the centroids
        targets: The target of the closest centroid for each sample
    """
    for i in range(number_of_clusters):
        diff = features - centroids[i]
        if weights is not None:
            # Weighted L2 distance with exponent beta
            dists[:, i] = np.sqrt(np.sum((weights ** beta) * (diff ** 2), axis=1))
        else:
            # Standard Euclidean distance
            dists[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))

    dists_min = np.min(dists, axis=1)
    targets = np.argmin(dists, axis=1)
    return dists_min, targets


def maximization_step(features: np.ndarray,
                      targets: np.ndarray,
                      number_of_clusters: int) -> np.ndarray:
    """
    Updates the centroids by computing the mean of all samples assigned to each cluster

    Parameters:
        features: The input data features
        targets: The targets indicating the cluster assignment of each sample
        number_of_clusters: The number of clusters

    Returns:
        centroids: The updated centroids
    """
    n_features = features.shape[1]
    centroids = np.zeros((number_of_clusters, n_features))
    for i in range(number_of_clusters):
        idx = np.where(targets == i)[0]
        if idx.size > 0:
            centroids[i] = np.mean(features[idx], axis=0)
    return centroids

def update_weights(features: np.ndarray,
                   targets: np.ndarray,
                   centroids: np.ndarray,
                   beta: float) -> np.ndarray:
    """
    Update feature weights for WK-means
    """
    n_features = features.shape[1]
    k = centroids.shape[0]
    Dv = np.zeros(n_features)

    # Calculate within-cluster variance for each feature
    for v in range(n_features):
        for i in range(k):
            idx = np.where(targets == i)[0]
            if idx.size > 0:
                Dv[v] += np.sum((features[idx, v] - centroids[i, v]) ** 2)

    # Avoid division by zero
    Dv = np.maximum(Dv, 1e-12)
    # Update weights
    weights = 1.0 / (Dv ** (1.0 / (beta - 1.0)))
    weights /= np.sum(weights)  # Normalize
    return weights


class KMeans():
    def __init__(self,
                 number_of_clusters: int = 1,
                 max_number_of_epochs: int = 20,
                 random_state: int = None,
                 n_init: int = 10,
                 use_weights: bool = False,
                 beta: float = 2.0) -> None:
        """
        K-Means or Weighted K-Means clustering.

        Parameters:
            number_of_clusters: Number of clusters 
            max_number_of_epochs: Maximum iterations
            random_state: Random seed
            n_init: Number of runs with different centroid seeds
            use_weights: If True, runs Weighted K-Means
            beta: Weight exponent parameter (WK-means)
        """
        self.k = number_of_clusters
        self.max_epochs = max_number_of_epochs
        self.random_state = random_state
        self.n_init = n_init
        self.use_weights = use_weights  
        self.beta = beta
        self.centroids = None
        self.feature_weights = None 

    def _init_centroids(self, features: np.ndarray) -> None:
        """
        Initializes centroids using the K-Means++ algorithm
        """
        n_samples = features.shape[0]
        centroids = np.empty((self.k, features.shape[1]), dtype=features.dtype)

        idx = np.random.randint(n_samples)
        centroids[0] = features[idx]

        for i in range(1, self.k):
            dist_sq = np.min(np.sum((features[:, np.newaxis, :] - centroids[:i]) ** 2, axis=2), axis=1)
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = features[next_idx]

        return centroids


    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fits the K-Means model to the input data

        Parameters:
            features: Feature matrix of the training data

        Returns:
            predictions: Cluster labels for each sample
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = features.shape[0]
        n_features = features.shape[1]
        best_loss = np.inf
        best_centroids = None
        best_targets = None
        best_weights = None

        for _ in range(self.n_init):
            centroids = self._init_centroids(features)
            weights = np.ones(n_features) / n_features if self.use_weights else None
            dists = np.zeros((n_samples, self.k))
            prev_loss = None

            for _ in range(self.max_epochs):
                dists_min, targets = expectation_step(features, centroids, dists, self.k, weights, self.beta)
                centroids = maximization_step(features, targets, self.k)

                if self.use_weights:
                    weights = update_weights(features, targets, centroids, self.beta)

                loss = np.sum(dists_min)
                if prev_loss is not None and np.isclose(loss, prev_loss, atol=1e-6):
                    break
                prev_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_centroids = centroids.copy()
                best_targets = targets.copy()
                best_weights = weights.copy() if self.use_weights else None

        self.centroids = best_centroids
        self.feature_weights = best_weights
        return best_targets
    
    def __str__(self) -> str:
        return "K-Means Clustering"
