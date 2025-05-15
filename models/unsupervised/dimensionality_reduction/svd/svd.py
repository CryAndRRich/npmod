import numpy as np

class SVD():
    def __init__(self, n_components: int = None) -> None:
        """
        Initialize truncated SVD model

        Parameters:
            n_components: Number of singular vectors to retain
        """
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.cum_explained_variance_ = None
        self.mean_ = None

    def fit(self, features: np.ndarray) -> None:
        """
        Compute truncated SVD on the input matrix

        Parameters:
            features: Input data matrix
        """
        X = features.astype(np.float64)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Full SVD decomposition
        _, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        # Determine number of components
        k = self.n_components or VT.shape[0]

        # Truncate
        self.components_ = VT[:k]
        self.singular_values_ = S[:k]

        # Compute explained variance ratio
        total_variance = np.sum(S**2)
        variances = (S**2)[:k]
        self.explained_variance_ratio_ = variances / total_variance
        self.cum_explained_variance_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Project data onto retained singular vectors

        Parameters:
            features: New data matrix 

        Returns:
            projection: Transformed data
        """
        X = features.astype(np.float64)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def __str__(self) -> str:
        return "Singular Value Decomposition (SVD)"
