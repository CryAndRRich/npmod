import numpy as np

class SpectralClustering():
    def __init__(self, 
                 n_clusters: int = 2, 
                 max_iterations: int = 100, 
                 laplacian_type: str = "normalized",
                 gamma: float = 1.0,
                 random_state: int = 42) -> None:
        """
        Spectral Clustering with vectorized similarity and safe Laplacian computation.

        Parameters:
            n_clusters: Number of clusters
            max_iterations: Max iterations for k-means
            laplacian_type: "unnormalized", "normalized", "symmetric"
            gamma: Scaling factor for RBF kernel
            random_state: Seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.laplacian_type = laplacian_type
        self.gamma = gamma
        self.random_state = random_state

    
    def _compute_similarity(self, features: np.ndarray) -> np.ndarray:
        """
        Computes the similarity matrix using negative squared Euclidean distance

        Parameters:
            features: Feature matrix of the input data

        Returns:
            W: The similarity matrix
        """
        sq_dists = np.sum((features[:, np.newaxis, :] - features[np.newaxis, :, :]) ** 2, axis=2)
        W = np.exp(-self.gamma * sq_dists)
        np.fill_diagonal(W, 0.0)
        return W

    def _compute_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Computes the graph Laplacian based on the selected type

        Parameters:
            W: The similarity matrix

        Returns:
            L: The Laplacian matrix
        """
        D = np.diag(W.sum(axis=1))
        if self.laplacian_type == "unnormalized":
            L = D - W
        elif self.laplacian_type == "normalized":
            D_inv = np.diag([1/d if d > 0 else 0.0 for d in np.diag(D)])
            L = np.eye(W.shape[0]) - D_inv @ W
        elif self.laplacian_type == "symmetric":
            D_inv_sqrt = np.diag([1/np.sqrt(d) if d > 0 else 0.0 for d in np.diag(D)])
            L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        else:
            raise ValueError(f"Unknown Laplacian type: {self.laplacian_type}")
        return L

    def _k_means(self, features: np.ndarray) -> np.ndarray:
        """
        Runs k-means clustering on the given features

        Parameters:
            features: The feature matrix after spectral embedding

        Returns:
            labels: Cluster labels for each sample
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = features.shape[0]
        centroids = features[rng.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iterations):
            dists = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            labels = dists.argmin(axis=1)

            new_centroids = np.array([
                features[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.n_clusters)
            ])

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fits the Spectral Clustering model to the input data

        Parameters:
            features: Feature matrix of the training data

        Returns:
            labels: Cluster labels for each sample
        """
        W = self._compute_similarity(features)
        L = self._compute_laplacian(W)

        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(L)
        idx = np.argsort(eigen_vals)[:self.n_clusters]  # k smallest
        X_spec = eigen_vecs[:, idx]

        # Row-normalize for symmetric Laplacian
        if self.laplacian_type == "symmetric":
            norms = np.linalg.norm(X_spec, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X_spec = X_spec / norms

        labels = self._k_means(X_spec)
        return labels

    def __str__(self) -> str:
        return "Spectral Clustering"
