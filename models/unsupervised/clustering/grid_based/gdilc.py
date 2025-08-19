from typing import Tuple
import numpy as np
from itertools import product
from collections import defaultdict

class UnionFind:
    """Union-Find (Disjoint Set) structure for efficient cluster merge"""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px

class GDILC:
    def __init__(self, 
                 m: int = 5, 
                 density_threshold: float = None, 
                 distance_threshold: float = None):
        """
        GDILC algorithm

        Parameters:
            m: Number of intervals per dimension for grid
            density_threshold: Minimum density to be considered a core point
            distance_threshold: Maximum distance to merge clusters
        """
        self.m = m
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.labels_ = None

    def _grid_index(self, point: np.ndarray) -> Tuple[int]:
        """Get the grid index of a point"""
        return tuple(np.minimum((point * self.m).astype(int), self.m - 1))

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit GDILC on data features and return labels for each point

        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels for each sample
        """
        n_samples, _ = features.shape
        self.labels_ = -1 * np.ones(n_samples, dtype=int)

        # assign points to grid cells
        grid = defaultdict(list)
        for i, point in enumerate(features):
            idx = self._grid_index(point)
            grid[idx].append(i)

        # compute distance threshold if not provided
        if self.distance_threshold is None:
            # median distance to nearest neighbor as threshold
            from scipy.spatial import distance_matrix
            dist_mat = distance_matrix(features, features)
            np.fill_diagonal(dist_mat, np.inf)
            nearest_dist = np.min(dist_mat, axis=1)
            self.distance_threshold = np.median(nearest_dist)

        # compute density vector for each point using neighbor cells
        densities = np.zeros(n_samples)
        for i, point in enumerate(features):
            idx = self._grid_index(point)
            # neighbor cells including itself
            neighbor_cells = product(*[[max(0, x-1), x, min(self.m-1, x+1)] for x in idx])
            neighbor_indices = []
            for cell in neighbor_cells:
                neighbor_indices.extend(grid.get(cell, []))
            dists = np.linalg.norm(features[neighbor_indices] - point, axis=1)
            densities[i] = np.sum(dists <= self.distance_threshold)

        # compute density threshold if not provided
        if self.density_threshold is None:
            self.density_threshold = np.mean(densities)

        # identify core points
        core_points = np.where(densities >= self.density_threshold)[0]

        # initialize union-find
        uf = UnionFind(n_samples)

        # merge clusters for points within distance threshold in neighbor cells
        for i in core_points:
            idx = self._grid_index(features[i])
            neighbor_cells = product(*[[max(0, x-1), x, min(self.m-1, x+1)] for x in idx])
            for cell in neighbor_cells:
                for j in grid.get(cell, []):
                    if densities[j] >= self.density_threshold and np.linalg.norm(features[i]-features[j]) <= self.distance_threshold:
                        uf.union(i, j)

        # assign labels based on union-find
        cluster_map = {}
        cluster_id = 0
        for i in range(n_samples):
            if densities[i] >= self.density_threshold:
                root = uf.find(i)
                if root not in cluster_map:
                    cluster_map[root] = cluster_id
                    cluster_id += 1
                self.labels_[i] = cluster_map[root]

        # mark non-core points as outliers (-1)
        self.labels_[densities < self.density_threshold] = -1

        return self.labels_

    def __str__(self):
        return "GDILC (Grid-Based Density-IsoLine Clustering)"
