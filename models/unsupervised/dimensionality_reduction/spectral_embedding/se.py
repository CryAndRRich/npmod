import numpy as np

class SpectralEmbedding():
    def __init__(self, 
                 n_components: int = 2, 
                 n_neighbors: int = 5, 
                 sigma: float = 1.0) -> None:
        """
        Initialize Spectral Embedding model

        Parameters:
            n_components: Number of dimensions for output embedding
            n_neighbors: Number of nearest neighbors to construct graph
            sigma: Gaussian kernel width for edge weights
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.sigma = sigma
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

    def _construct_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct affinity matrix using k-nearest neighbors and Gaussian kernel

        Parameters:
            X: Input data matrix 

        Returns:
            W: Affinity matrix 
        """
        n_samples = X.shape[0]
        dist = self._pairwise_distances(X)
        idx = np.argsort(dist, axis=1)[:, 1:self.n_neighbors+1]

        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in idx[i]:
                weight = np.exp(-dist[i, j]**2 / (2 * self.sigma**2))
                W[i, j] = weight
                W[j, i] = weight
        return W

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the Spectral Embedding model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        n_samples = X.shape[0]

        # Construct affinity matrix W
        W = self._construct_affinity_matrix(X)

        # Compute normalized symmetric Laplacian
        degree = np.sum(W, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-12))
        L = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt

        # Eigen-decomposition
        eigen_vals, eigen_vectors = np.linalg.eigh(L)
        idx = np.argsort(eigen_vals)[1:self.n_components+1]
        self.embedding_ = eigen_vectors[:, idx]
        return self.embedding_

    def __str__(self) -> str:
        return "Spectral Embedding"
