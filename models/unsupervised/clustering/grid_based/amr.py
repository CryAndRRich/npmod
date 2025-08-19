import numpy as np
from typing import List, Tuple

class AMRNode():
    def __init__(self, 
                 bounds: List[Tuple[float, float]], 
                 points_idx: List[int], 
                 level: int = 0) -> None:
        """
        Node of the AMR tree

        Parameters:
            bounds: List of (min, max) tuples for each dimension
            points_idx: Indices of points in this cell
            level: Level in the hierarchy (0=root)
        """
        self.bounds = bounds
        self.points_idx = points_idx
        self.level = level
        self.children = []
        self.is_leaf = True
        self.cluster_id = None

class AMR():
    def __init__(self, 
                 max_points_per_cell: int = 50, 
                 max_level: int = 5):
        """
        Adaptive Mesh Refinement clustering

        Parameters:
            max_points_per_cell: Density threshold for refining a cell
            max_level: Maximum refinement level
        """
        self.max_points_per_cell = max_points_per_cell
        self.max_level = max_level
        self.root = None
        self.labels_ = None
        self.cluster_count = 0

    def _subdivide_bounds(self, 
                          bounds: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Subdivide the current bounds into 2^d child cells
        """
        # For each dimension, create lower and upper half
        splits = []
        for low, high in bounds:
            mid = (low + high) / 2
            splits.append([(low, mid), (mid, high)])

        # Cartesian product to generate all child bounds
        from itertools import product
        child_bounds = [list(b) for b in product(*splits)]
        return child_bounds

    def _build_tree(self, 
                    points: np.ndarray, 
                    node: AMRNode) -> None:
        """
        Recursively refine cells with too many points
        """
        if len(node.points_idx) <= self.max_points_per_cell or node.level >= self.max_level:
            node.is_leaf = True
            return

        node.is_leaf = False
        child_bounds_list = self._subdivide_bounds(node.bounds)

        for child_bounds in child_bounds_list:
            # Find points inside child cell
            inside_idx = []
            for idx in node.points_idx:
                pt = points[idx]
                if all(low <= pt[d] < high for d, (low, high) in enumerate(child_bounds)):
                    inside_idx.append(idx)
            if inside_idx:
                child_node = AMRNode(bounds=child_bounds, points_idx=inside_idx, level=node.level + 1)
                node.children.append(child_node)
                self._build_tree(points, child_node)

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit AMR clustering to the data

        Parameters:
            features: Data matrix (n_samples, n_features)

        Returns:
            labels_: Cluster assignments for each point
        """
        n_samples, n_features = features.shape
        self.labels_ = np.full(n_samples, -1, dtype=int)
        # Initial root bounds
        bounds = [(features[:, d].min(), features[:, d].max()) for d in range(n_features)]
        self.root = AMRNode(bounds=bounds, points_idx=list(range(n_samples)))
        self._build_tree(features, self.root)

        # Assign clusters from leaves
        self.cluster_count = 0
        self._assign_clusters(self.root)
        return self.labels_

    def _assign_clusters(self, node: AMRNode) -> None:
        """
        Recursively assign cluster IDs from leaves to points
        """
        if node.is_leaf:
            node.cluster_id = self.cluster_count
            for idx in node.points_idx:
                self.labels_[idx] = node.cluster_id
            self.cluster_count += 1
        else:
            for child in node.children:
                self._assign_clusters(child)

    def __str__(self) -> str:
        return "AMR (Adaptive Mesh Refinement Clustering)"
