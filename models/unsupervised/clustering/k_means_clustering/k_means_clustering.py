from typing import Tuple
import numpy as np

def expectation_step(features: np.ndarray,
                     centroids: np.ndarray,
                     dists: np.ndarray,
                     number_of_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the distance from each feature to each centroid and 
    assigns each feature to the closest centroid

    Parameters:
        features: The input data features
        centroids: The current centroids of the clusters
        dists: Preallocated array to store distances from each sample to each centroid
        number_of_clusters: The number of clusters

    Returns:
        dists_min: The minimum distance of each sample to the centroids
        targets: The target of the closest centroid for each sample
    """
    for i in range(number_of_clusters):
        # Compute Euclidean distance
        diff = features - centroids[i]
        dists[:, i] = np.sqrt(np.sum(diff ** 2, axis=1))

    dists_min = np.min(dists, axis=1)
    targets = np.argmin(dists, axis=1)
    return dists_min, targets


def maximization_step(features: np.ndarray,
                      centroids: np.ndarray,
                      targets: np.ndarray,
                      number_of_clusters: int) -> np.ndarray:
    """
    Updates the centroids by computing the mean of all samples assigned to each cluster

    Parameters:
        features: The input data features
        centroids: The current centroids of the clusters (to be updated)
        targets: The targets indicating the cluster assignment of each sample
        number_of_clusters: The number of clusters

    Returns:
        centroids: The updated centroids
    """
    for i in range(number_of_clusters):
        idxs = np.where(targets == i)[0]
        if idxs.size > 0:
            centroids[i] = np.mean(features[idxs], axis=0)
    return centroids

class KMeans():
    def __init__(self,
                 number_of_clusters: int = 1,
                 max_number_of_epochs: int = 20,
                 random_state: int = None) -> None:
        """
        K-Means Clustering model for unsupervised learning

        Parameters:
            number_of_clusters: The number of clusters to form
            max_number_of_epochs: The maximum number of iterations to run the algorithm
            random_state: Seed for centroid initialization
        """
        self.k = number_of_clusters
        self.max_epochs = max_number_of_epochs
        self.random_state = random_state
        self.centroids = None

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

        n_samples, n_features = features.shape
        # Initialize centroids randomly within data bounds
        feature_min = features.min(axis=0)
        feature_max = features.max(axis=0)
        self.centroids = feature_min + (feature_max - feature_min) * np.random.rand(self.k, n_features)

        dists = np.zeros((n_samples, self.k))
        prev_loss = None

        for epoch in range(1, self.max_epochs + 1):
            # E-step
            dists_min, targets = expectation_step(features, self.centroids, dists, self.k)
            # M-step
            self.centroids = maximization_step(features, self.centroids, targets, self.k)

            loss = np.sum(dists_min)
            if prev_loss is not None and loss == prev_loss:
                print(f"Converged at epoch {epoch}/{self.max_epochs}")
                break
            prev_loss = loss
        else:
            print(f"Reached max epochs {self.max_epochs}/{self.max_epochs}")

        return targets

    def __str__(self) -> str:
        return "K-Means Clustering"
