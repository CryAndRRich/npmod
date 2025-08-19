import numpy as np
from typing import Optional, List

class IKMeans:
    def __init__(self,
                 max_clusters: Optional[int] = None,
                 threshold: int = 3,
                 max_number_of_epochs: int = 300):
        
        self.max_clusters = max_clusters - 1 # Labels start from 0, so we subtract 1
        self.threshold = threshold
        self.max_number_of_epochs = max_number_of_epochs
        self.centroids = None
        self.labels_ = None

    def _center_of_gravity(self, features: np.ndarray) -> np.ndarray:
        """Compute the centroid of the given features"""
        return np.mean(features, axis=0)

    def _farthest_point(self, features: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Find the point farthest from the given centroid"""
        distances = np.linalg.norm(features - centroid, axis=1)
        farthest_idx = np.argmax(distances)
        return features[farthest_idx]

    def _assign_clusters(self,
                         features: np.ndarray,
                         centroid_a: np.ndarray,
                         centroid_b: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest of two centroids"""
        dist_a = np.linalg.norm(features - centroid_a, axis=1)
        dist_b = np.linalg.norm(features - centroid_b, axis=1)
        return np.where(dist_b < dist_a, 1, 0)

    def fit(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        labels = np.full(n_samples, -1)
        cluster_id = 0
        centroids_list: List[np.ndarray] = []

        # Initial centroid: center of gravity
        cg = self._center_of_gravity(features)
        remaining_points = features.copy()

        while True:
            # Find farthest point as new centroid
            c = self._farthest_point(remaining_points, cg)

            # Assign points between cg and c
            assign_mask = self._assign_clusters(remaining_points, cg, c) == 1
            cluster_points = remaining_points[assign_mask]

            if len(cluster_points) < self.threshold:
                # Small cluster discarded
                break

            sg = np.mean(cluster_points, axis=0)  # New centroid for anomalous cluster

            # Assign labels for these cluster points
            for point in cluster_points:
                idx = np.where((features == point).all(axis=1))[0]
                labels[idx] = cluster_id

            centroids_list.append(sg)
            cluster_id += 1

            # Update cg to sg for next iteration
            cg = sg
            remaining_points = features[labels == -1]

            # Stopping conditions
            if len(remaining_points) == 0:
                break
            if self.max_clusters is not None and cluster_id >= self.max_clusters:
                break

        self.centroids = np.array(centroids_list)
        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "IK-Means Clustering"
