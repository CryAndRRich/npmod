import numpy as np
from typing import Tuple, Optional

class KMedians:
    def __init__(self,
                 number_of_clusters: int = 3,
                 max_number_of_epochs: int = 300,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None) -> None:
        
        self.k = number_of_clusters
        self.max_epochs = max_number_of_epochs
        self.tol = tol
        self.random_state = random_state
        self.medians = None
        self.labels_ = None

    def _init_medians(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        return features[indices]

    def _total_cost(self,
                    features: np.ndarray,
                    medians: np.ndarray) -> Tuple[float, np.ndarray]:
        
        dists = np.sum(np.abs(features[:, None, :] - medians[None, :, :]), axis=2)  # L1 norm
        min_dists = np.min(dists, axis=1)
        labels = np.argmin(dists, axis=1)
        return np.sum(min_dists), labels

    def fit(self, features: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.medians = self._init_medians(features)
        best_cost, labels = self._total_cost(features, self.medians)

        for _ in range(self.max_epochs):
            new_medians = np.copy(self.medians)
            for k in range(self.k):
                cluster_points = features[labels == k]
                if len(cluster_points) > 0:
                    new_medians[k] = np.median(cluster_points, axis=0)
            
            cost, new_labels = self._total_cost(features, new_medians)
            shift = np.sum(np.abs(new_medians - self.medians))
            self.medians = new_medians
            labels = new_labels

            if shift < self.tol or cost >= best_cost:
                break
            best_cost = cost

        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "K-Medians Clustering"
