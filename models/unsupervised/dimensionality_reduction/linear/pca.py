import numpy as np

class PCA():
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None

    def fit(self, features: np.ndarray) -> None:
        """
        Compute the principal components from input features
        
        Parameters
            features: Input data matrix 
        """
        # Standardize data (z-score normalization)
        features = features.copy()
        self.mean = np.mean(features, axis=0)
        features_centered = features - self.mean

        # Eigen-decomposition of covariance matrix
        cov_mat = np.cov(features_centered.T)
        eigen_vals, eigen_vectors = np.linalg.eigh(cov_mat)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals_sorted = eigen_vals[idx]
        eigen_vectors_sorted = eigen_vectors[:, idx].T

        # Adjust eigenvector signs for consistency
        max_abs_idx = np.argmax(np.abs(eigen_vectors_sorted), axis=1)
        signs = np.sign(eigen_vectors_sorted[np.arange(len(max_abs_idx)), max_abs_idx])
        eigen_vectors_sorted = eigen_vectors_sorted * signs[:, np.newaxis]

        # Select top n_components
        self.components = eigen_vectors_sorted[: self.n_components, :]

        # Explained variance ratio
        total_variance = np.sum(eigen_vals_sorted)
        self.explained_variance_ratio = eigen_vals_sorted[: self.n_components] / total_variance
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Project input features onto principal component axes

        Parameters:
            features: Input data matrix 
        Returns
            projection: Transformed data 
        """
        if self.components is None:
            raise ValueError("PCA model is not fitted yet. Call fit() first.")

        features = features.copy()
        features_std = features - self.mean
        projection = features_std.dot(self.components.T)

        return projection

    def inverse_transform(self, projection: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from its PCA projection back to original feature space

        Parameters
            projection : Transformed data in the reduced-dimensional PCA space

        Returns
            reconstructed : Approximate reconstruction of the original data
        """
        if self.components is None:
            raise ValueError("PCA model is not fitted yet. Call fit() first.")
        
        reconstructed = projection.dot(self.components) + self.mean
        return reconstructed
    
    def __str__(self) -> str:
        return "PCA (Principal Component Analysis)"
