from typing import Optional
import numpy as np

class AffinityPropagation():
    def __init__(self,
                 max_iterations: int = 200,
                 preference: Optional[float] = None) -> None:
        """
        Affinity Propagation clustering

        Parameters:
            max_iterations: Maximum number of iterations for message passing
            preference: Preference value for exemplar selection (diagonal of similarity matrix).
                        If None, set to median of similarity values
        """
        self.max_iterations = max_iterations
        self.preference = preference
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
        # Compute similarity matrix: negative squared Euclidean distance
        similarity_mat = -np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
        # Set preference
        if self.preference is None:
            pref = np.median(similarity_mat)
        else:
            pref = self.preference

        np.fill_diagonal(similarity_mat, pref)
        # Initialize responsibility and availability
        R = np.zeros_like(similarity_mat)
        A = np.zeros_like(similarity_mat)
        # Message passing
        for _ in range(self.max_iterations):
            # Update responsibilities
            AS = A + similarity_mat
            max_vals = np.max(AS, axis=1, keepdims=True)
            second_max = np.max(np.where(AS == max_vals, -np.inf, AS), axis=1, keepdims=True)
            R = similarity_mat - np.where(AS == max_vals, second_max, max_vals)
            # Update availabilities
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, R.diagonal())
            A = np.minimum(0, Rp.sum(axis=0, keepdims=True) - Rp)
            A[np.diag_indices(n_samples)] = Rp.sum(axis=0)

        # Identify exemplars
        E = A + R
        exemplars = np.where(np.diag(E) > 0)[0]
        if exemplars.size == 0:
            # Fallback: choose one exemplar
            exemplars = np.array([np.argmax(np.diag(E))])
        # Assign labels
        labels = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            # Choose exemplar that maximizes similarity_mat[i, k]
            labels[i] = exemplars[np.argmax(similarity_mat[i, exemplars])]
        # Relabel to consecutive ints
        unique_ex = np.unique(labels)
        mapping = {ex: idx for idx, ex in enumerate(unique_ex)}
        labels = np.array([mapping[l] for l in labels])
        self.labels = labels
        return labels

    def __str__(self) -> str:
        return "Affinity Propagation"
