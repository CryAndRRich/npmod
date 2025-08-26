import numpy as np

class LDA():
    def __init__(self, n_components: int = None) -> None:
        """
        Initialize Linear Discriminant Analysis reducer

        Parameters:
            n_components: Number of discriminant components to keep
        """
        self.n_components = n_components
        self.scalings = None
        self.means = None
        self.class_means = None
        self.explained_variance_ratio = None

    def fit(self, 
            features: np.ndarray, 
            targets: np.ndarray) -> None:
        """
        Fit LDA model by computing projection matrix

        Parameters:
            features: Input data matrix 
            targets: Class labels vector 
        """
        X = features.astype(np.float64)
        y = np.asarray(targets)
        classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = classes.shape[0]

        max_components = n_classes - 1
        k = self.n_components or max_components
        if k > max_components:
            raise ValueError(f"n_components must be <= {max_components}")

        self.means = np.mean(X, axis=0)
        self.class_means = {c: X[y == c].mean(axis=0) for c in classes}

        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in classes:
            Xc = X[y == c]
            mean_diff = (self.class_means[c] - self.means)[:, None]
            SW += (Xc - self.class_means[c]).T @ (Xc - self.class_means[c])
            SB += Xc.shape[0] * (mean_diff @ mean_diff.T)

        # Solve eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW) @ SB)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        self.scalings = eigvecs[:, :k]
        self.scalings /= np.linalg.norm(self.scalings, axis=0)

        total = np.sum(eigvals)
        self.explained_variance_ratio = eigvals[:k] / total

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Project data onto discriminant components

        Parameters:
            features: Input data matrix 

        Returns:
            np.ndarray: Transformed data matrix
        """
        if self.scalings is None:
            raise ValueError("LDA model is not fitted yet. Call fit() first.")
        return features.astype(np.float64) @ self.scalings

    def __str__(self) -> str:
        return "LDA (Linear Discriminant Analysis)"
