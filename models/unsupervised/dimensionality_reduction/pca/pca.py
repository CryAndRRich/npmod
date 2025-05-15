import numpy as np

class PCA():
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def fit(self, features: np.ndarray) -> None:
        """
        Compute the principal components from input features
        
        Parameters
            features: Input data matrix 
        """
        # Standardize data
        features = features.copy()
        self.mean = np.mean(features, axis=0)
        self.scale = np.std(features, axis=0)
        features_std = (features - self.mean) / self.scale

        # Eigen-decomposition of covariance matrix
        cov_mat = np.cov(features_std.T)
        eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)

        # Adjust eigenvector signs for consistency
        max_abs_idx = np.argmax(np.abs(eigen_vectors), axis=0)
        signs = np.sign(eigen_vectors[max_abs_idx, range(eigen_vectors.shape[0])])
        eigen_vectors = eigen_vectors * signs[np.newaxis, :]
        eigen_vectors = eigen_vectors.T

        # Sort eigenvalues and eigenvectors in descending order
        eigen_pairs = [
            (np.abs(eigen_vals[i]), eigen_vectors[i, :])
            for i in range(len(eigen_vals))
        ]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        eigen_vals_sorted = np.array([ev for ev, _ in eigen_pairs])
        eigen_vectors_sorted = np.array([vec for _, vec in eigen_pairs])

        # Select top n_components
        self.components = eigen_vectors_sorted[: self.n_components, :]

        # Explained variance ratio and cumulative
        total_variance = np.sum(eigen_vals)
        self.explained_variance_ratio = [val / total_variance for val in eigen_vals_sorted[: self.n_components]]
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Project input features onto principal component axes

        Parameters:
            features: Input data matrix 
        Returns
            projection: Transformed data 
        """
        features = features.copy()
        features_std = (features - self.mean) / self.scale
        projection = features_std.dot(self.components.T)

        return projection

    def __str__(self) -> str:
        return "Principal Component Analysis (PCA)"
