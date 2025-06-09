from typing import List
import numpy as np

class AgglomerativeClustering():
    def __init__(self,
                 number_of_clusters: int = 2,
                 linkage: str = "single") -> None:
        """
        Agglomerative (hierarchical) clustering using specified linkage criteria

        Parameters:
            number_of_clusters: The number of clusters to form
            linkage: Linkage criterion to use - one of "single", "complete", "avg", "centroid"
        """
        if linkage not in {"single", "complete", "avg", "centroid"}:
            raise ValueError(f"Unsupported linkage: {linkage}")
        self.k = number_of_clusters
        self.linkage = linkage
        self.labels = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Performs hierarchical agglomerative clustering on the data

        Parameters:
            features: Feature matrix of the training data

        Returns:
            labels: Cluster labels for each sample (0 to k-1)
        """
        n_samples = features.shape[0]
        # Initialize each sample as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # Precompute pairwise distances
        def cluster_distance(c1: List[int], 
                             c2: List[int]) -> float:
            pts1 = features[c1]
            pts2 = features[c2]
            dists = np.sqrt(((pts1[:, None, :] - pts2[None, :, :]) ** 2).sum(axis=2))
            if self.linkage == "single":
                return float(np.min(dists))
            elif self.linkage == "complete":
                return float(np.max(dists))
            elif self.linkage == "avg":
                return float(np.mean(dists))
            # Centroid
            center1 = pts1.mean(axis=0)
            center2 = pts2.mean(axis=0)
            return float(np.linalg.norm(center1 - center2))

        # Iteratively merge until desired number of clusters
        while len(clusters) > self.k:
            n_clusters = len(clusters)
            min_dist = np.inf
            pair_to_merge = (0, 1)
            # find closest pair
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    dist = cluster_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        pair_to_merge = (i, j)
            i, j = pair_to_merge
            # Merge j into i and remove j
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        # Assign labels based on clusters
        labels = np.empty(n_samples, dtype=int)
        for cluster_id, members in enumerate(clusters):
            for idx in members:
                labels[idx] = cluster_id
        self.labels = labels
        return labels

    def __str__(self) -> str:
        return "Agglomerative Clustering"
