import numpy as np
from typing import List, Optional, Tuple
from itertools import product


class STINGCell:
    def __init__(self,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 level: int,
                 parent: Optional["STINGCell"] = None) -> None:
        """Single cell in STING hierarchical grid"""

        self.bounds = bounds
        self.level = level
        self.parent = parent
        self.children = []

        self.n = 0
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.point_ids = []

        # cluster id assigned later
        self.cluster_id = None


class STING:
    def __init__(self,
                 max_level: int = 3,
                 min_cell_size: Optional[float] = None,
                 min_points: int = 1) -> None:
        """
        STING: Statistical Information Grid-based Clustering

        Parameters:
            max_level: Maximum depth of hierarchical grid 
            min_cell_size: Minimum size of cell (stop splitting if smaller)
            min_points: Minimum number of points per cell to be considered dense
        """
        self.max_level = max_level
        self.min_cell_size = min_cell_size
        self.min_points = min_points

        self.root_ = None
        self.dim_ = None
        self.labels_ = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit STING on data features and return labels for each point

        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels for each sample
        """
        X = np.asarray(features, dtype=float)
        n, d = X.shape
        self.dim_ = d

        mins = X.min(axis=0)
        maxs = X.max(axis=0) + 1e-8
        self.root_ = STINGCell((mins, maxs), level=0)

        # recursive build
        self._build_recursive(self.root_, X, np.arange(n), level=0)

        # query dense cells
        dense_cells = self.query_relevant_cells(threshold=self.min_points)

        # build adjacency graph of dense cells
        adj = {i: [] for i in range(len(dense_cells))}
        for i, ci in enumerate(dense_cells):
            for j, cj in enumerate(dense_cells):
                if i < j and self._is_adjacent(ci, cj):
                    adj[i].append(j)
                    adj[j].append(i)

        # BFS/DFS to merge connected dense cells
        labels = -np.ones(n, dtype=int)
        visited = set()
        cid = 0
        for i in range(len(dense_cells)):
            if i in visited:
                continue
            stack = [i]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                dense_cells[u].cluster_id = cid
                for pid in dense_cells[u].point_ids:
                    labels[pid] = cid
                stack.extend(adj[u])
            cid += 1

        self.labels_ = labels
        return self.labels_

    def _build_recursive(self,
                         cell: STINGCell,
                         X: np.ndarray,
                         ids: np.ndarray,
                         level: int) -> None:

        pts = X[ids]
        cell.point_ids = ids.tolist()
        cell.n = len(ids)

        if len(ids) > 0:
            cell.mean = pts.mean(axis=0)
            cell.std = pts.std(axis=0)
            cell.min = pts.min(axis=0)
            cell.max = pts.max(axis=0)

        # stop condition
        if level >= self.max_level:
            return
        if self.min_cell_size is not None:
            if np.any(cell.bounds[1] - cell.bounds[0] <= self.min_cell_size):
                return
        if len(ids) <= self.min_points:
            return

        # split into 2^d children
        mins, maxs = cell.bounds
        mid = (mins + maxs) / 2.0

        for mask in product([0, 1], repeat=self.dim_):
            cmin, cmax = [], []
            for j in range(self.dim_):
                if mask[j] == 0:
                    cmin.append(mins[j])
                    cmax.append(mid[j])
                else:
                    cmin.append(mid[j])
                    cmax.append(maxs[j])
            cmin, cmax = np.array(cmin), np.array(cmax)

            sel = np.all((X[ids] >= cmin) & (X[ids] < cmax), axis=1)
            cid = ids[sel]
            if len(cid) == 0:
                continue
            child = STINGCell((cmin, cmax), level=level+1, parent=cell)
            cell.children.append(child)
            self._build_recursive(child, X, cid, level+1)

    def query_relevant_cells(self, threshold: int) -> List[STINGCell]:
        """Return all leaf cells with >= threshold points"""

        if self.root_ is None:
            raise ValueError("Must fit before query")
        out = []
        self._collect_dense(self.root_, threshold, out)
        return out

    def _collect_dense(self, cell: STINGCell, threshold: int, out: List[STINGCell]) -> None:
        if not cell.children:  # leaf
            if cell.n >= threshold:
                out.append(cell)
        else:
            for ch in cell.children:
                self._collect_dense(ch, threshold, out)

    def _is_adjacent(self, c1: STINGCell, c2: STINGCell) -> bool:
        """Check if two cells are spatially adjacent (touch or overlap)"""
        
        for d in range(self.dim_):
            if c1.bounds[1][d] < c2.bounds[0][d] or c2.bounds[1][d] < c1.bounds[0][d]:
                return False
        return True

    def __str__(self) -> str:
        return "STING (Statistical Information Grid)"
