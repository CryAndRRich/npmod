from typing import Optional
import numpy as np

class AffinityPropagation():
    def __init__(self,
                 max_iterations: int = 200,
                 damping: float = 0.5,
                 preference: Optional[float] = None,
                 convergence_iter: int = 15) -> None:
        """
        Affinity Propagation clustering

        Parameters:
            max_iterations: Maximum number of iterations for message passing
            damping: Damping factor (0.5~0.9) to avoid oscillations
            preference: Preference value for exemplars. If None, set to median similarity
            convergence_iter: Number of iterations to check convergence
        """
        self.max_iterations = max_iterations
        self.damping = damping
        self.preference = preference
        self.convergence_iter = convergence_iter
        self.labels = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fits the model to the input data

        Parameters:
            features: Feature matrix 

        Returns:
            labels: Cluster labels for each sample
        """
        n_samples = features.shape[0]

        # Compute similarity matrix (negative squared Euclidean distance)
        S = -np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)

        # Set preference (diagonal)
        if self.preference is None:
            pref = np.median(S)
        else:
            pref = self.preference
        np.fill_diagonal(S, pref)

        # Initialize responsibility and availability matrices
        R = np.zeros_like(S)
        A = np.zeros_like(S)

        # For convergence check
        exemplars_history = np.zeros((self.convergence_iter, n_samples), dtype=int)

        for it in range(self.max_iterations):
            # ---- Update Responsibilities ----
            AS = A + S
            max_idx = np.argmax(AS, axis=1)
            max_val = AS[np.arange(n_samples), max_idx]

            AS[np.arange(n_samples), max_idx] = -np.inf
            second_max_val = AS.max(axis=1)

            R_new = S - max_val[:, None]
            R_new[np.arange(n_samples), max_idx] = S[np.arange(n_samples), max_idx] - second_max_val

            R = self.damping * R + (1 - self.damping) * R_new

            # ---- Update Availabilities ----
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, R.diagonal())
            A_new = np.zeros_like(A)
            for k in range(n_samples):
                A_new[:, k] = np.minimum(0, Rp.sum(axis=0) - Rp[:, k])
            np.fill_diagonal(A_new, Rp.sum(axis=0))
            A = self.damping * A + (1 - self.damping) * A_new

            # ---- Check convergence ----
            E = A + R
            exemplars = np.where(np.diag(E) > 0)[0]
            if exemplars.size == 0:
                exemplars = np.array([np.argmax(np.diag(E))])
            labels = np.array([exemplars[np.argmax(S[i, exemplars])] for i in range(n_samples)])
            exemplars_history[it % self.convergence_iter] = labels

            if it >= self.convergence_iter:
                if np.all(exemplars_history[0] == exemplars_history).all():
                    break  # converged

        # Remap labels to consecutive integers
        unique_labels = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([mapping[l] for l in labels])
        self.labels = labels
        return labels

    def __str__(self) -> str:
        return "Affinity Propagation"
