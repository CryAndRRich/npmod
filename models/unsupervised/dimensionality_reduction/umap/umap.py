from typing import Tuple
import numpy as np

class UMAP():
    def __init__(self, 
                 n_neighbors: int = 15, 
                 min_dist: float = 0.1, 
                 n_components: int = 2, 
                 n_epochs: int = 200, 
                 learning_rate: float = 1.0, 
                 random_state: int = 42) -> None:
        """
        Initialize UMAP reducer with hyperparameters

        Parameters:
            n_neighbors: Number of nearest neighbors to use for local approximations
            min_dist: Minimum distance between points in low-dimensional space
            n_components: Target dimension of the embedding
            n_epochs: Number of optimization epochs
            learning_rate: Step size for embedding optimization
            random_state: Seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.embedding_ = None
        self._X_fit = None
        self._P = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances between all pairs of samples

        Parameters:
            X: Data matrix

        Returns:
            np.ndarray: Matrix distances between all pairs of samples
        """
        sum_X = np.sum(np.square(X), axis=1)
        return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]

    def _smooth_knn_dist(self, 
                         knn_dists: np.ndarray, 
                         tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local connectivity (rho) and bandwidth (sigma) per point via binary search

        Parameters:
            knn_dists: Distances to k-th nearest neighbors
            tol: Tolerance for binary search convergence

        Returns:
            rho: Local connectivity thresholds
            sigma: Bandwidth for each point
        """
        n_samples, k = knn_dists.shape
        rho = knn_dists[:, 0]
        sigma = np.zeros(n_samples)
        target = np.log2(k)
        for i in range(n_samples):
            lo, hi = 0.0, np.inf
            mid = 1.0
            di = knn_dists[i]
            for _ in range(50):
                ps = np.exp(-(di - rho[i]) / mid)
                sum_ps = np.sum(ps)
                if abs(sum_ps - target) < tol:
                    break
                if sum_ps > target:
                    lo = mid
                    mid = mid * 2 if hi == np.inf else (mid + hi) / 2
                else:
                    hi = mid
                    mid = mid / 2 if lo == 0 else (mid + lo) / 2
            sigma[i] = mid
        return rho, sigma

    def _compute_probability_graph(self, X: np.ndarray) -> np.ndarray:
        """
        Build fuzzy simplicial set (probability graph) P

        Parameters:
            X: Input data matrix

        Returns:
            P: Symmetric joint probability matrix
        """
        n_samples = X.shape[0]
        distances = self._pairwise_distances(X)
        knn_indices = np.argsort(distances, axis=1)[:, 1:self.n_neighbors+1]
        knn_dists = np.take_along_axis(distances, knn_indices, axis=1)
        rho, sigma = self._smooth_knn_dist(knn_dists)
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for idx, dist in zip(knn_indices[i], knn_dists[i]):
                P[i, idx] = np.exp(-(dist - rho[i]) / sigma[i])
        P = (P + P.T) - P * P.T  # Fuzzy union
        return P

    def fit(self, features: np.ndarray) -> None:
        """
        Fit UMAP to the input data and compute low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            self: Fitted UMAP instance with embedding_ attribute
        """
        np.random.seed(self.random_state)
        X = features.astype(np.float64)
        self._X_fit = X
        # Compute probability graph
        self._P = self._compute_probability_graph(X)
        # Initialize embedding randomly
        n_samples = X.shape[0]
        Y = np.random.randn(n_samples, self.n_components)
        # Optimization: simple attractive forces only
        for _ in range(self.n_epochs):
            for i in range(n_samples):
                # Attractive force toward neighbors
                neighbors = np.where(self._P[i] > 0)[0]
                if len(neighbors) == 0:
                    continue
                diff = Y[neighbors] - Y[i]
                weights = self._P[i, neighbors][:, None]
                Y[i] += self.learning_rate * np.sum(weights * diff, axis=0)
        self.embedding_ = Y

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform new data into existing UMAP embedding space via weighted average of neighbors

        Parameters:
            features: New data matrix of shape (m_samples, n_features)

        Returns:
            embedding: Transformed data of shape (m_samples, n_components)
        """
        if self.embedding_ is None or self._X_fit is None or self._P is None:
            raise ValueError("UMAP model has not been fitted yet. Call fit() before transform().")
        X_new = features.astype(np.float64)
        # Compute distances to training data
        distances = -2 * np.dot(X_new, self._X_fit.T) + np.sum(X_new**2, axis=1)[:, None] + np.sum(self._X_fit**2, axis=1)[None, :]
        knn_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        knn_dists = np.take_along_axis(distances, knn_indices, axis=1)
        rho, sigma = self._smooth_knn_dist(knn_dists)
        weights = np.exp(-(knn_dists - rho[:, None]) / sigma[:, None])
        weights /= np.sum(weights, axis=1)[:, None]
        return weights.dot(self.embedding_)

    def __str__(self) -> str:
        return "Uniform Manifold Approximation and Projection (UMAP)"
