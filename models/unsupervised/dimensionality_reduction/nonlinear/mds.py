import numpy as np

class MDS():
    def __init__(self, 
                 n_components: int = 2, 
                 n_init: int = 4, 
                 max_iter: int = 300, 
                 eps: float = 1e-3, 
                 random_state: int = 42) -> None:
        """
        Multidimensional Scaling (MDS) using SMACOF

        Parameters:
            n_components: target embedding dimension
            n_init: number of restarts
            max_iter: maximum number of iterations
            eps: convergence tolerance on stress
            random_state: reproducibility
        """
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = np.random.RandomState(random_state)

        self.embedding_ = None
        self.stress_ = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        sum_X = np.sum(X ** 2, axis=1)
        dist2 = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
        return np.sqrt(np.maximum(dist2, 1e-12))

    def _smacof_single(self, D: np.ndarray, X_init: np.ndarray) -> tuple:
        """One run of SMACOF"""
        n_samples = D.shape[0]
        X = X_init.copy()

        for it in range(self.max_iter):
            dist_X = self._pairwise_distances(X)
            # Avoid divide by zero
            mask = dist_X > 0
            B = np.zeros((n_samples, n_samples))
            B[mask] = -D[mask] / dist_X[mask]
            B[np.arange(n_samples), np.arange(n_samples)] = -np.sum(B, axis=1)

            X_new = (1.0 / n_samples) * B.dot(X)
            stress = np.sum((D[mask] - dist_X[mask])**2)

            # Convergence check
            if it > 0 and abs(self.stress_ - stress) < self.eps:
                break

            X = X_new
            self.stress_ = stress

        return X, self.stress_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the MDS model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        D = self._pairwise_distances(X)

        best_stress = np.inf
        best_X = None

        for _ in range(self.n_init):
            X_init = self.random_state.normal(scale=1e-4, size=(X.shape[0], self.n_components))
            X_out, stress = self._smacof_single(D, X_init)
            if stress < best_stress:
                best_stress = stress
                best_X = X_out

        self.embedding_ = best_X
        self.stress_ = best_stress
        return self.embedding_

    def __str__(self) -> str:
        return "MDS (Multidimensional Scaling)"
