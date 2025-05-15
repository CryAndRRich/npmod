import numpy as np

class ISOMAP():
    def __init__(self, 
                 n_neighbors: int = 5, 
                 n_components: int = 2) -> None:
        """
        Initialize ISOMAP model with hyperparameters

        Parameters:
            n_neighbors: Number of nearest neighbors for graph construction
            n_components: Target dimension for embedding
        """
        self.n_neighbors = n_neighbors
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
        Fit the ISOMAP model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        n_samples = X.shape[0]

        # Compute full distance matrix
        dist = self._pairwise_distances(X)

        # Construct neighborhood graph
        graph = np.full((n_samples, n_samples), np.inf)
        for i in range(n_samples):
            neighbors = np.argsort(dist[i])[:self.n_neighbors + 1]
            for j in neighbors:
                graph[i, j] = dist[i, j]
        np.fill_diagonal(graph, 0.0)

        # Compute shortest paths (Floyd-Warshall)
        D = graph.copy()
        for k in range(n_samples):
            D = np.minimum(D, D[:, k, None] + D[None, k, :])

        # Apply classical MDS to geodesic distances
        # Centering matrix
        J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        # Squared distances
        D2 = D**2
        B = -0.5 * J.dot(D2).dot(J)

        # Eigen-decomposition
        eigen_vals, eigen_vectors = np.linalg.eigh(B)
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vectors = eigen_vectors[:, idx]

        # Select top components
        lambdas = np.maximum(eigen_vals[:self.n_components], 0)
        vectors = eigen_vectors[:, :self.n_components]
        self.embedding_ = vectors * np.sqrt(lambdas)[None, :]
        return self.embedding_

    def __str__(self) -> str:
        return "Isometric Mapping (ISOMAP)"
