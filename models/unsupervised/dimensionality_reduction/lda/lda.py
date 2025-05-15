import numpy as np

class LDA():
    def __init__(self, n_components: int = None) -> None:
        """
        Initialize Linear Discriminant Analysis reducer

        Parameters:
            n_components: Number of discriminant components to keep
        """
        self.n_components = n_components
        self.scalings_ = None
        self.means_ = None
        self.class_means_ = None
        self.explained_variance_ratio_ = None

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fit LDA model by computing projection matrix

        Parameters:
            features: Input data matrix 
            targets: Class labels vector 
        """
        # Ensure inputs
        X = features.astype(np.float64)
        y = np.asarray(targets)
        classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = classes.shape[0]

        # Determine number of components
        max_components = n_classes - 1
        k = self.n_components or max_components
        if k > max_components:
            raise ValueError(f"n_components must be <= {max_components}")

        # Compute overall mean and class means
        self.means_ = np.mean(X, axis=0)
        self.class_means_ = {c: X[y == c].mean(axis=0) for c in classes}

        # Compute within-class scatter SW and between-class scatter SB
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in classes:
            Xc = X[y == c]
            mean_diff = (self.class_means_[c] - self.means_)[:, None]
            SW += np.dot((Xc - self.class_means_[c]).T, (Xc - self.class_means_[c]))
            SB += Xc.shape[0] * mean_diff.dot(mean_diff.T)

        # Solve generalized eigenvalue problem for SW^{-1} SB
        eigen_vals, eigen_vectors = np.linalg.eig(np.linalg.pinv(SW).dot(SB))
        # Sort eigenvectors by descending eigenvalue magnitude
        sorted_idx = np.argsort(np.abs(eigen_vals))[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        eigen_vals = eigen_vals[sorted_idx]

        # Select top k components
        self.scalings_ = eigen_vectors[:, :k].real

        # Compute explained variance ratio
        total = np.sum(np.abs(eigen_vals))
        self.explained_variance_ratio_ = np.abs(eigen_vals[:k]) / total

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Project data onto discriminant components

        Parameters:
            features: Input data matrix 

        Returns:
            np.ndarray: Transformed data matrix
        """
        if self.scalings_ is None:
            raise ValueError("LDA model is not fitted yet. Call fit() before transform().")
        X = features.astype(np.float64)
        return np.dot(X, self.scalings_)

    def __str__(self) -> str:
        return "Linear Discriminant Analysis (LDA)"
