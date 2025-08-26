import numpy as np

class SpectralEmbedding():
    def __init__(self, 
                 n_components: int = 2, 
                 n_neighbors: int = 5, 
                 sigma: float = 1.0, 
                 affinity: str = "nearest_neighbors") -> None:
        """
        Spectral Embedding (Laplacian Eigenmaps)

        Parameters:
            n_components: Output dimension
            n_neighbors: k for kNN (used if affinity="nearest_neighbors")
            sigma: Kernel width (used if affinity="rbf")
            affinity: "nearest_neighbors" or "rbf"
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.affinity = affinity
        self.embedding_ = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        sum_X = np.sum(X ** 2, axis=1)
        dist2 = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
        return np.sqrt(np.maximum(dist2, 0))
    
    def _largest_connected_component(self, graph: np.ndarray) -> np.ndarray:
        """Find mask for largest connected component using BFS"""
        n = graph.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = []
        
        for start in range(n):
            if not visited[start]:
                queue = [start]
                comp = []
                visited[start] = True
                while queue:
                    u = queue.pop()
                    comp.append(u)
                    neighbors = np.where(graph[u] < np.inf)[0]
                    for v in neighbors:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                components.append(comp)

        largest = max(components, key=len)
        return np.array(sorted(largest))

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

        if self.affinity == "nearest_neighbors":
            idx = np.argsort(dist, axis=1)[:, 1:self.n_neighbors+1]
            W = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in idx[i]:
                    W[i, j] = 1.0
                    W[j, i] = 1.0
        elif self.affinity == "rbf":
            W = np.exp(-dist**2 / (2 * self.sigma**2))
            np.fill_diagonal(W, 0.0)
        else:
            raise ValueError("affinity must be 'nearest_neighbors' or 'rbf'")
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

        # Build affinity matrix
        W = self._construct_affinity_matrix(X)

        # Keep only largest connected component
        idx = self._largest_connected_component(W)
        W = W[np.ix_(idx, idx)]

        # Degree and normalized Laplacian
        degree = np.sum(W, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-12))
        L = np.eye(len(idx)) - D_inv_sqrt @ W @ D_inv_sqrt

        # Eigen-decomposition
        eigen_vals, eigen_vectors = np.linalg.eigh(L)
        sorted_idx = np.argsort(eigen_vals)

        chosen = sorted_idx[1:self.n_components+1]
        embedding = eigen_vectors[:, chosen]

        self.embedding_ = np.zeros((n_samples, self.n_components))
        self.embedding_[idx] = embedding
        return self.embedding_
    
    def __str__(self) -> str:
        return "Spectral Embedding"
