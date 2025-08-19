from typing import List, Tuple, Optional
import numpy as np
import heapq


class OPTICS():
    def __init__(self,
                 max_eps: float = np.inf,
                 min_samples: int = 5,
                 eps: float = 0.5) -> None:
        """
        OPTICS - Ordering Points To Identify the Clustering Structure

        Parameters:
            max_eps: Generating distance (maximum neighborhood radius)
            min_samples: Minimum number of samples to define a core point
            eps: Epsilon for DBSCAN-equivalent extraction (not used in OPTICS itself, but for convenience)
        """
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if max_eps <= 0 and not np.isinf(max_eps):
            raise ValueError("max_eps must be > 0 or np.inf")

        self.max_eps = max_eps
        self.min_samples = int(min_samples)
        self.eps = eps

        self.reachability_ = None     
        self.core_distances_ = None  
        self.ordering_ = None       

    def _region_query(self, 
                      features: np.ndarray, 
                      idx: int) -> np.ndarray:
        """Return indices of all points within max_eps of features[idx]"""
        diffs = features - features[idx]
        dists = np.linalg.norm(diffs, axis=1)
        if np.isinf(self.max_eps):
            return np.arange(features.shape[0], dtype=int)
        return np.where(dists <= self.max_eps)[0]

    def _core_distance(self,
                       features: np.ndarray,
                       neighbors: np.ndarray,
                       point_idx: int) -> Optional[float]:
        """Core-distance = distance to the min_samples-th nearest neighbor within the neighborhood"""
        if neighbors.size < self.min_samples:
            return None
        # Distances to neighbors (including possibly itself at distance 0)
        dists = np.linalg.norm(features[neighbors] - features[point_idx], axis=1)
        dists.sort()
        # The min_samples-th nearest neighbor (1-based) → index min_samples-1 (0-based)
        return float(dists[self.min_samples - 1])

    def _update_seeds(self,
                      point_idx: int,
                      neighbors: np.ndarray,
                      features: np.ndarray,
                      processed: np.ndarray,
                      reachability: np.ndarray,
                      core_dist: float,
                      seeds: List[Tuple[float, int]]) -> None:
        """Update reachability distances of neighbors using the given core-distance"""

        p = features[point_idx]
        # Compute distances only for unprocessed neighbors
        unproc_mask = ~processed[neighbors]
        if not np.any(unproc_mask):
            return
        nbrs = neighbors[unproc_mask]
        dists = np.linalg.norm(features[nbrs] - p, axis=1)

        new_reach = np.maximum(core_dist, dists)
        # Update if better
        for nbr, r in zip(nbrs, new_reach):
            if r < reachability[nbr]:
                reachability[nbr] = r
                heapq.heappush(seeds, (r, nbr))

    def fit(self, features: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Run OPTICS to compute the ordering and reachability distances then 
        extract a flat clustering equivalent to running DBSCAN(eps, min_samples)
        using the OPTICS ordering and reachability plot

        Parameters:
            features: feature matrix

        Returns:
            labels: Cluster labels in {0..K-1}, noise = -1
        """
        X = np.asarray(features, dtype=float)
        n = X.shape[0]
        if n == 0:
            self.reachability_ = np.array([], dtype=float)
            self.core_distances_ = np.array([], dtype=float)
            self.ordering_ = []
            return [], self.reachability_

        reachability = np.full(n, np.inf, dtype=float)
        core_distances = np.full(n, np.nan, dtype=float)
        processed = np.zeros(n, dtype=bool)
        ordering = []

        for point in range(n):
            if processed[point]:
                continue

            # Start a new component
            neighbors = self._region_query(X, point)
            processed[point] = True
            ordering.append(point)

            core_dist = self._core_distance(X, neighbors, point)
            if core_dist is not None:
                core_distances[point] = core_dist

                # Initialize seed heap with neighbors whose reachability can be improved
                seeds = []
                self._update_seeds(point, neighbors, X, processed, reachability, core_dist, seeds)

                # Expand while seeds available
                while seeds:
                    _, q = heapq.heappop(seeds)
                    if processed[q]:
                        continue
                    neighbors_q = self._region_query(X, q)
                    processed[q] = True
                    ordering.append(q)

                    core_dist_q = self._core_distance(X, neighbors_q, q)
                    if core_dist_q is not None:
                        core_distances[q] = core_dist_q
                        self._update_seeds(q, neighbors_q, X, processed, reachability, core_dist_q, seeds)
            else:
                # Not a core point; just move on (reachability remains inf)
                pass

        self.reachability_ = reachability
        self.core_distances_ = core_distances
        self.ordering_ = ordering

        labels = -np.ones(len(self.ordering_), dtype=int)  # temporary array in ordering-space

        cluster_id = -1
        for i, p in enumerate(self.ordering_):
            r = self.reachability_[p]
            c = self.core_distances_[p]
            if (r > self.eps) and (not np.isnan(c)) and (c <= self.eps):
                # start a new cluster
                cluster_id += 1
                labels[i] = cluster_id
            elif r <= self.eps:
                # continue current cluster if any
                if cluster_id >= 0:
                    labels[i] = cluster_id
                # else remains -1 (noise between clusters)
            else:
                # r > eps and (c is NaN or c > eps) -> noise unless later absorbed
                pass

        # Map labels from ordering-space back to original sample indices
        out = -np.ones(len(self.ordering_), dtype=int)
        for i, p in enumerate(self.ordering_):
            out[p] = labels[i]
            
        return out

    def __str__(self) -> str:
        return "OPTICS (Ordering Points To Identify the Clustering Structure)"
