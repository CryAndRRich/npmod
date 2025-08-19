from typing import List, Tuple
import numpy as np
from collections import defaultdict

class DivisiveClustering():
    def __init__(self, 
                 number_of_clusters: int = None, 
                 zahn: bool = False, 
                 inconsistency_factor: float = 1.5) -> None:
        """
        Minimum Spanning Tree (MST) based Divisive clustering with K-cut or Zahn's inconsistency pruning

        Parameters:
            number_of_clusters: Desired number of clusters. If None and zahn=True, clustering is determined by inconsistency pruning
            zahn : If True, uses Zahn's inconsistency-based pruning instead of fixed K-cut
            inconsistency_factor : Factor threshold to determine if an edge is inconsistent in Zahn's algorithm
        """
        self.n_clusters = number_of_clusters
        self.zahn = zahn
        self.factor = inconsistency_factor
        self.labels_ = None
        self.edges_ = None  # (u, v, weight) 

    def _pairwise_distances(self, features: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Compute all pairwise edges with Euclidean distance

        Parameters
            features: The input data points
        """
        n = features.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(features[i] - features[j])
                edges.append((i, j, d))
        return edges

    def _kruskal_mst(self, 
                     edges: List[Tuple[int, int, float]], 
                     n: int) -> List[Tuple[int, int, float]]:
        """
        Build the MST using Kruskal's algorithm

        Parameters:
            edges: Edges in the form (u, v, weight), unsorted
            n: Number of nodes
        """
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            xr, yr = find(x), find(y)
            if xr == yr:
                return False
            if rank[xr] < rank[yr]:
                parent[xr] = yr
            elif rank[xr] > rank[yr]:
                parent[yr] = xr
            else:
                parent[yr] = xr
                rank[xr] += 1
            return True

        mst_edges = []
        for u, v, w in sorted(edges, key=lambda e: e[2]):
            if union(u, v):
                mst_edges.append((u, v, w))
            if len(mst_edges) == n - 1:
                break
        return mst_edges

    def _zahn_inconsistency_pruning(self, mst_edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Perform Zahn's inconsistency-based edge removal

        Parameters:
            mst_edges: Edges in MST 
        """
        adj = defaultdict(list)
        for u, v, w in mst_edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        edges_set = set(mst_edges)
        removed = True
        while removed:
            removed = False
            for u, v, w in list(edges_set):
                # Compute neighborhood average weight
                neigh_weights = [wt for _, wt in adj[u]] + [wt for _, wt in adj[v]]
                avg_w = np.mean(neigh_weights) if neigh_weights else 0
                if avg_w > 0 and w > self.factor * avg_w:
                    # Remove edge
                    edges_set.remove((u, v, w))
                    adj[u] = [(x, wt) for x, wt in adj[u] if x != v]
                    adj[v] = [(x, wt) for x, wt in adj[v] if x != u]
                    removed = True
        return list(edges_set)

    def fit(self, features: np.ndarray):
        """
        Fit the MST-based clustering on input data

        Parameters
            features: The input data points
        """
        n = features.shape[0]
        if self.n_clusters is not None and (self.n_clusters < 1 or self.n_clusters > n):
            raise ValueError("n_clusters must be between 1 and n_samples")

        # Build MST using Kruskal
        all_edges = self._pairwise_distances(features)
        mst_edges = self._kruskal_mst(all_edges, n)
        self.edges_ = mst_edges

        if self.zahn:
            # Zahn's pruning
            pruned_edges = self._zahn_inconsistency_pruning(mst_edges)
        else:
            # Fixed K-cut
            edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)
            remove_edges = set(edges_sorted[:self.n_clusters - 1])
            pruned_edges = [e for e in mst_edges if e not in remove_edges]

        # Build clusters from remaining edges
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            xr, yr = find(x), find(y)
            if xr != yr:
                parent[yr] = xr

        for u, v, _ in pruned_edges:
            union(u, v)

        # Assign labels
        root_map = {}
        label = 0
        labels = np.empty(n, dtype=int)
        for i in range(n):
            root = find(i)
            if root not in root_map:
                root_map[root] = label
                label += 1
            labels[i] = root_map[root]

        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "Divisive Clustering"
