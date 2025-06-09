import numpy as np

class SpectralClustering():
    def __init__(self, 
                 number_of_clusters: int = 2, 
                 max_iterations: int = 100, 
                 laplacian_type: str = "unnormalized") -> None:
        """
        Spectral Clustering model using graph Laplacian and eigen decomposition

        Parameters:
            number_of_clusters: The number of clusters to form
            max_iterations: The maximum number of iterations for k-means
            laplacian_type: The type of Laplacian ("unnormalized", "normalized", "symmetric")
        """
        self.k = number_of_clusters
        self.max_iterations = max_iterations
        self.laplacian_type = laplacian_type
    
    def _compute_similarity(self, features: np.ndarray) -> np.ndarray:
        """
        Computes the similarity matrix using negative squared Euclidean distance

        Parameters:
            features: Feature matrix of the input data

        Returns:
            W: The similarity matrix
        """
        n_samples = features.shape[0]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(features[i] - features[j]) ** 2
                W[i, j] = W[j, i] = np.exp(-dist)  # RBF kernel form (optional scaling)
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
            D_inv = np.linalg.inv(D)
            L = np.eye(W.shape[0]) - D_inv @ W
        elif self.laplacian_type == "symmetric":
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal()))
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
        n_samples = features.shape[0]
        rng = np.random.default_rng(seed=42)
        centroids = features[rng.choice(n_samples, size=self.k, replace=False)]

        for _ in range(self.max_iterations):
            dists = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            labels = dists.argmin(axis=1)

            new_centroids = np.array([
                features[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.k)
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

        _, eigen_vectors = np.linalg.eigh(L)
        X_spec = eigen_vectors[:, :self.k]  # Take k smallest eigenvectors

        # Normalize rows if using symmetric normalized Laplacian
        if self.laplacian_type == "symmetric":
            norms = np.linalg.norm(X_spec, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X_spec = X_spec / norms

        labels = self._k_means(X_spec)
        return labels

    def __str__(self) -> str:
        return "Spectral Clustering"
