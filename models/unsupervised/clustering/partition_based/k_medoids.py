from typing import Tuple
import numpy as np

class KMedoids:
    def __init__(self,
                 number_of_clusters: int = 1,
                 max_number_of_epochs: int = 20,
                 random_state: int = None) -> None:
        
        self.k = number_of_clusters
        self.max_epochs = max_number_of_epochs
        self.random_state = random_state
        self.medoids = None
        self.labels_ = None

    def _init_medoids(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        return indices 

    def _total_cost(self, 
                    features: np.ndarray, 
                    medoid_indices: np.ndarray) -> Tuple[float, np.ndarray]:
        
        medoids = features[medoid_indices]
        dists = np.linalg.norm(features[:, None, :] - medoids[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        labels = np.argmin(dists, axis=1)
        return np.sum(min_dists), labels

    def fit(self, features: np.ndarray) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = features.shape[0]
        medoid_indices = self._init_medoids(features)
        best_cost, labels = self._total_cost(features, medoid_indices)

        for _ in range(self.max_epochs):
            improved = False
            for mi in medoid_indices: 
                for xi in range(n_samples):  
                    if xi in medoid_indices:
                        continue
                    new_medoids = medoid_indices.copy()
                    new_medoids[new_medoids == mi] = xi 
                    cost, new_labels = self._total_cost(features, new_medoids)
                    if cost < best_cost:
                        medoid_indices = new_medoids
                        best_cost = cost
                        labels = new_labels
                        improved = True
            if not improved:
                break

        self.medoids = features[medoid_indices]
        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "K-Medoids Clustering"
