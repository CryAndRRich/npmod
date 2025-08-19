import math
from typing import List, Optional, Tuple
import numpy as np

def _get_neighbor_offsets(d: int, 
                          moore: bool = True) -> List[Tuple[int, ...]]:
    """Return neighbor offsets for d-dim grid"""
    if moore:
        ranges = [(-1, 0, 1)] * d
        offs = []
        from itertools import product
        for comb in product(*ranges):
            if all(v == 0 for v in comb):
                continue
            offs.append(tuple(comb))
        return offs
    else:
        offs = []
        for axis in range(d):
            v = [0] * d
            v[axis] = 1
            offs.append(tuple(v))
            v2 = [0] * d
            v2[axis] = -1
            offs.append(tuple(v2))
        return offs


def _compute_grid_bounds(X: np.ndarray,
                         grid_shape: Optional[Tuple[int, ...]] = None,
                         cell_size: Optional[Tuple[float, ...]] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, ...]]:
    """Compute grid min/max per dimension and bin edges"""
    
    n, d = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    eps = 1e-8
    maxs = maxs + eps

    if grid_shape is not None and cell_size is not None:
        raise ValueError("Provide either grid_shape or cell_size, not both.")

    if grid_shape is None and cell_size is None:
        nbins = int(math.ceil(n ** (1.0 / d)))
        grid_shape = tuple([nbins] * d)

    if grid_shape is not None:
        # grid_shape: number of bins per dimension
        bin_edges = []
        for i in range(d):
            edges = np.linspace(mins[i], maxs[i], num=grid_shape[i] + 1)
            bin_edges.append(edges)
        return (mins, maxs), tuple(bin_edges)

    # cell_size provided
    bin_edges = []
    for i in range(d):
        si = cell_size[i]
        if si <= 0:
            raise ValueError("cell_size must be positive")
        edges = np.arange(mins[i], maxs[i] + si, si)
        if edges[-1] < maxs[i]:
            edges = np.append(edges, maxs[i] + eps)
        bin_edges.append(edges)
    return (mins, maxs), tuple(bin_edges)


def _point_to_block_index(point: np.ndarray, 
                          bin_edges: Tuple[np.ndarray, ...]) -> Tuple[int, ...]:
    """Map a single point to block indices (tuple)"""

    inds = []
    for dim, edges in enumerate(bin_edges):
        # searchsorted returns insertion index; subtract 1 to get left bin
        i = np.searchsorted(edges, point[dim], side="right") - 1
        # clip to valid range
        i = max(0, min(i, len(edges) - 2))
        inds.append(int(i))
    return tuple(inds)

def _block_volume(bin_edges: Tuple[np.ndarray, ...], 
                  idx: Tuple[int, ...]) -> float:
    v = 1.0
    for dim, edges in enumerate(bin_edges):
        a = edges[idx[dim]]
        b = edges[idx[dim] + 1]
        v *= (b - a)
    return float(v)


class GRIDCLUS():
    def __init__(self,
                 grid_shape: Optional[int | Tuple[int, ...]] = None,
                 cell_size: Optional[float | Tuple[float, ...]] = None,
                 min_points: int = 1,
                 neighborhood: str = "moore",
                 assign_sparse_to_neighbor: bool = True,) -> None:
        """
        GRIDCLUS (grid-based clustering)

        Parameters:
            grid_shape: Number of bins per dimension
            cell_size: Alternatively, specify cell size per dimension
            min_points: Minimum number of points in a block to consider it "dense"
            neighborhood: "moore" (full surrounding neighbors) or "von_neumann" (orthogonal neighbors)
            assign_sparse_to_neighbor: If True, points in sparse blocks are assigned to adjacent dense-cluster if any; 
                                       otherwise marked as noise (-1)
        """
        self.grid_shape = grid_shape
        self.cell_size = cell_size
        self.min_points = int(min_points)
        if neighborhood not in {"moore", "von_neumann"}:
            raise ValueError('neighborhood must be one of {"moore","von_neumann"}')
        self.neighborhood = neighborhood
        self.assign_sparse_to_neighbor = bool(assign_sparse_to_neighbor)

        self.labels_ = None
        self.block_map_ = None  
        self.block_density_ = None
        self.block_count_ = None
        self.cluster_blocks_ = None
        self.bin_edges_ = None
        self.grid_bounds_ = None

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit GRIDCLUS on data features and return labels for each point

        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels for each sample (-1 indicates noise)
        """
        X = np.asarray(features, dtype=float)
        n, d = X.shape

        # Normalize grid parameters
        if self.grid_shape is not None:
            if isinstance(self.grid_shape, int):
                grid_shape = tuple([int(self.grid_shape)] * d)
            else:
                grid_shape = tuple(int(x) for x in self.grid_shape)
        else:
            grid_shape = None

        if self.cell_size is not None:
            if isinstance(self.cell_size, (int, float)):
                cell_size = tuple([float(self.cell_size)] * d)
            else:
                cell_size = tuple(float(x) for x in self.cell_size)
        else:
            cell_size = None

        # Build grid bin edges
        (mins, maxs), bin_edges = _compute_grid_bounds(X, grid_shape=grid_shape, cell_size=cell_size)
        self.grid_bounds_ = (mins, maxs)
        self.bin_edges_ = bin_edges

        # Assign points to blocks
        block_map = {}
        for i in range(n):
            idx = _point_to_block_index(X[i], bin_edges)
            block_map.setdefault(idx, []).append(i)
        self.block_map_ = block_map

        # Compute counts and densities
        block_count = {}
        block_density = {}
        for idx, pts in block_map.items():
            cnt = len(pts)
            vol = _block_volume(bin_edges, idx)
            dens = cnt / vol if vol > 0 else float("inf")
            block_count[idx] = cnt
            block_density[idx] = dens
        self.block_count_ = block_count
        self.block_density_ = block_density

        # Identify dense blocks
        dense_blocks = set(idx for idx, cnt in block_count.items() if cnt >= self.min_points)

        # Build neighbor offsets
        offs = _get_neighbor_offsets(d, moore=(self.neighborhood == "moore"))

        def neighbors_of(idx_t: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            out = []
            for off in offs:
                nb = tuple(idx_t[j] + off[j] for j in range(d))
                if nb in block_map:
                    out.append(nb)
            return out

        # Cluster dense blocks by connected components (BFS/DFS)
        cluster_blocks = {}
        assigned_block = {}
        cluster_id = 0
        visited_blocks = set()

        for b in dense_blocks:
            if b in visited_blocks:
                continue
            # BFS to gather connected dense blocks
            stack = [b]
            visited_blocks.add(b)
            assigned_block[b] = cluster_id
            cluster_blocks[cluster_id] = [b]
            while stack:
                cur = stack.pop()
                for nb in neighbors_of(cur):
                    if nb in dense_blocks and nb not in visited_blocks:
                        visited_blocks.add(nb)
                        assigned_block[nb] = cluster_id
                        cluster_blocks[cluster_id].append(nb)
                        stack.append(nb)
            cluster_id += 1

        # Assign points in dense blocks to clusters (one-to-one)
        labels = -np.ones(n, dtype=int)
        for cid, blocks in cluster_blocks.items():
            for b in blocks:
                for pid in block_map[b]:
                    labels[pid] = cid

        # Optionally assign sparse block points to neighbor clusters
        if self.assign_sparse_to_neighbor:
            for b, pts in block_map.items():
                if b in assigned_block:
                    continue  # already assigned (dense)
                # check neighbor blocks for assigned clusters
                nb_keys = neighbors_of(b)
                neighbor_cluster_ids = []
                for nb in nb_keys:
                    if nb in assigned_block:
                        neighbor_cluster_ids.append(assigned_block[nb])
                if neighbor_cluster_ids:
                    # Assign to most common neighbor cluster
                    vals, counts = np.unique(neighbor_cluster_ids, return_counts=True)
                    best = vals[np.argmax(counts)]
                    for pid in pts:
                        labels[pid] = int(best)
                else:
                    pass

        self.cluster_blocks_ = cluster_blocks
        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "GRIDCLUS (Grid-based Clustering)"
