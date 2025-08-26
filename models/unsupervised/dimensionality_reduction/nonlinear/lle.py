import numpy as np

class LLE():
    def __init__(self, 
                 n_neighbors: int = 10, 
                 n_components: int = 2, 
                 reg: float = 1e-3) -> None:
        """
        Initialize Locally Linear Embedding (LLE) reducer

        Parameters:
            n_neighbors: Number of nearest neighbors to use for reconstruction
            n_components: Target dimension for embedding
            reg: Regularization constant to stabilize weight computation
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
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
        Fit the LLE model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        n_samples = X.shape[0]

        # Compute distances and find neighbors
        dist = self._pairwise_distances(X)
        neighbors = np.argsort(dist, axis=1)[:, 1:self.n_neighbors+1]

        # Compute reconstruction weights
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            Z = X[neighbors[i]] - X[i] 
            C = Z.dot(Z.T)  # Local covariance (k, k)
            C += np.eye(self.n_neighbors) * self.reg * np.trace(C)  # Regularization
            w = np.linalg.solve(C, np.ones(self.n_neighbors))
            w /= np.sum(w)
            W[i, neighbors[i]] = w

        # Construct cost matrix M = (I - W)'(I - W)
        I = np.eye(n_samples)
        M = (I - W).T.dot(I - W)

        # Eigen-decomposition of M
        eigen_vals, eigen_vectors = np.linalg.eigh(M)
        # Skip the smallest eigenvalue (0) and take next n_components
        idx = np.argsort(eigen_vals)[1:self.n_components+1]
        embedding = eigen_vectors[:, idx]

        self.embedding_ = embedding
        return embedding

    def __str__(self) -> str:
        return "LLE (Locally Linear Embedding)"
