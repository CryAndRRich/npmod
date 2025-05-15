import numpy as np

class MDS():
    def __init__(self, n_components: int = 2) -> None:
        """
        Initialize Multidimensional Scaling (MDS) reducer

        Parameters:
            n_components: Target dimension for embedding
        """
        self.n_components = n_components
        self.embedding_ = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance matrix

        Parameters:
            X: Data matrix 

        Returns:
            dist: Distance matrix 
        """
        sum_X = np.sum(X**2, axis=1)
        dist2 = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
        return np.sqrt(np.maximum(dist2, 0))

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the MDS model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation of shape (n_samples, n_components)
        """
        X = features.astype(np.float64)
        n_samples = X.shape[0]

        # Compute distance matrix
        dist = self._pairwise_distances(X)

        # Double center distances
        dist_squared = dist**2
        J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        B = -0.5 * J.dot(dist_squared).dot(J)

        # Eigen-decomposition
        eigen_vals, eigen_vectors = np.linalg.eigh(B)
        idx = np.argsort(eigen_vals)[::-1][:self.n_components]
        L = np.sqrt(np.maximum(eigen_vals[idx], 0))
        V = eigen_vectors[:, idx]

        # Compute embedding
        self.embedding_ = V * L[None, :]
        return self.embedding_

    def __str__(self) -> str:
        return "Multidimensional Scaling (MDS)"
