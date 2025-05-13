import numpy as np
import heapq
from typing import List, Tuple, Optional
from ....base import Model

class OPTICS(Model):
    def __init__(self,
                 eps: float = np.inf,
                 min_samples: int = 5) -> None:
        """
        Ordering Points To Identify the Clustering Structure

        Parameters:
            eps: Maximum neighborhood radius
            min_samples: Minimum number of samples to form a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.reachability = None
        self.ordering = None

    def _region_query(self, 
                      features: np.ndarray, 
                      idx: int) -> np.ndarray:
        """
        Finds all points within eps distance of the given point

        Parameters:
            features: Feature matrix 
            idx: Index of the query point

        Returns:
            neighbors: Array of indices of neighboring points within eps
        """
        diffs = features - features[idx]
        dists = np.linalg.norm(diffs, axis=1)
        return np.where(dists <= self.eps)[0]

    def _core_distance(self, 
                       features: np.ndarray, 
                       neighbors: np.ndarray, 
                       point_idx: int) -> Optional[float]:
        """
        Computes the core distance for a point: the distance to its min_samples-th nearest neighbor

        Parameters:
            features: Feature matrix (shape: [n_samples, n_features])
            neighbors: Array of neighbor indices for the point
            point_idx: Index of the point to compute core distance for

        Returns:
            core_dist: Core distance value, or None if point is not a core point
        """
        if neighbors.size < self.min_samples:
            return None
        # Distances from point to its neighbors
        dists = np.linalg.norm(features[neighbors] - features[point_idx], axis=1)
        sorted_dists = np.sort(dists)
        return float(sorted_dists[self.min_samples - 1])

    def fit(self, features: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Runs OPTICS algorithm to compute ordering and reachability distances

        Parameters:
            features: Feature matrix

        Returns:
            ordering: List of point indices in the order processed
            reachability: Array of reachability distances for each point
        """
        n_samples = features.shape[0]
        self.reachability = np.full(n_samples, np.inf)
        processed = np.zeros(n_samples, dtype=bool)
        ordering = []

        for point in range(n_samples):
            if processed[point]:
                continue
            # Compute neighbors and core distance
            neighbors = self._region_query(features, point)
            processed[point] = True
            ordering.append(point)
            core_dist = None
            if neighbors.size >= self.min_samples:
                # Distances of neighbors from point
                dists = np.linalg.norm(features[neighbors] - features[point], axis=1)
                core_dist = np.sort(dists)[self.min_samples-1]

            # Seeds priority queue
            if core_dist is not None:
                seeds = []
                for nbr in neighbors:
                    if processed[nbr]:
                        continue
                    new_reach = max(core_dist, np.linalg.norm(features[nbr] - features[point]))
                    if new_reach < self.reachability[nbr]:
                        self.reachability[nbr] = new_reach
                        heapq.heappush(seeds, (new_reach, nbr))

                # Expand cluster order
                while seeds:
                    _, q = heapq.heappop(seeds)
                    if processed[q]:
                        continue
                    processed[q] = True
                    ordering.append(q)
                    neighbors_q = self._region_query(features, q)
                    if neighbors_q.size >= self.min_samples:
                        dists_q = np.linalg.norm(features[neighbors_q] - features[q], axis=1)
                        core_dist_q = np.sort(dists_q)[self.min_samples-1]
                        for nbr in neighbors_q:
                            if processed[nbr]:
                                continue
                            new_reach = max(core_dist_q, np.linalg.norm(features[nbr] - features[q]))
                            if new_reach < self.reachability[nbr]:
                                self.reachability[nbr] = new_reach
                                heapq.heappush(seeds, (new_reach, nbr))
        self.ordering = ordering

        return ordering, self.reachability

    def __str__(self) -> str:
        return "Ordering Points To Identify the Clustering Structure (OPTICS)"
