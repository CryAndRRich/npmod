import numpy as np
from typing import List
from ....base import Model

def region_query(features: np.ndarray,
                 point_idx: int,
                 eps: float) -> List[int]:
    """
    Finds all points in the dataset within eps distance of the given point

    Parameters:
        features: Feature matrix 
        point_idx: Index of the point to query neighbors for
        eps: Neighborhood radius

    Returns:
        neighbors: List of indices of neighboring points
    """
    diffs = features - features[point_idx]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    neighbors = list(np.where(dists <= eps)[0])
    return neighbors

def expand_cluster(features: np.ndarray,
                   labels: np.ndarray,
                   point_idx: int,
                   cluster_id: int,
                   eps: float,
                   min_samples: int) -> bool:
    """
    Attempts to grow a new cluster with given seed point

    Parameters:
        features: Feature matrix 
        labels: Array of current labels for each point (-1 means unvisited, 0 means noise)
        point_idx: Index of the seed point for this cluster
        cluster_id: Current cluster label (starting from 1)
        eps: Neighborhood radius
        min_samples: Minimum number of points to form a dense region

    Returns:
        success: True if cluster was expanded, False if marked as noise
    """
    seeds = region_query(features, point_idx, eps)
    if len(seeds) < min_samples:
        labels[point_idx] = 0  # noise
        return False

    labels[seeds] = cluster_id
    seeds.remove(point_idx)

    while seeds:
        current = seeds.pop(0)
        result = region_query(features, current, eps)
        if len(result) >= min_samples:
            for idx in result:
                if labels[idx] in (-1, 0):
                    if labels[idx] == -1:
                        seeds.append(idx)
                    labels[idx] = cluster_id
    return True

class DBSCAN(Model):
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5) -> None:
        """
        Density-Based Spatial Clustering of Applications with Noise

        Parameters:
            eps: Neighborhood radius
            min_samples: Minimum number of points to form a dense region
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fits the DBSCAN model to the input data

        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels for each sample (-1 indicates noise)
        """
        n_samples = features.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for idx in range(n_samples):
            if labels[idx] != -1:
                continue
            if expand_cluster(features, labels, idx, cluster_id + 1, self.eps, self.min_samples):
                cluster_id += 1
        self.labels = labels
        return labels

    def __str__(self) -> str:
        return "Density-Based Spatial Clustering of Applications (DBSCAN)"
