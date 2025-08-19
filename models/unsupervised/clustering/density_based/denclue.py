from typing import Dict, List, Optional, Tuple
import numpy as np


def _euclidean_dist(a: np.ndarray, 
                    b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _gaussian_influence(dists: np.ndarray, 
                        sigma: float) -> np.ndarray:
    """
    Gaussian influence function: f_B(x, y) = exp(-||x - y||^2 / (2 sigma^2))

    Parameters:
        dists: distances ||x - y_i|| 
        sigma: bandwidth (> 0)

    Returns:
        weights: influence values 
    """
    s2 = (sigma * sigma) * 2.0
    val = np.exp(-np.clip((dists ** 2) / s2, 0.0, 80.0))
    return val

def _square_influence(dists: np.ndarray, 
                      sigma: float) -> np.ndarray:
    """
    Square wave influence function:
        f_B(x, y) = 1 if ||x - y|| <= sigma else 0

    Parameters:
        dists: distances ||x - y_i|| 
        sigma: radius (> 0)
    
    Returns:
        weights: 0/1 
    """
    return (dists <= sigma).astype(float)


def _density_and_gradient_at(x: np.ndarray,
                             data: np.ndarray,
                             sigma: float,
                             influence: str) -> Tuple[float, np.ndarray]:
    """Compute DENCLUE density and (unnormalized) gradient at a query point x"""

    diffs = data - x  
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))  

    if influence == "gaussian":
        w = _gaussian_influence(dists, sigma)  
    elif influence == "square":
        w = _square_influence(dists, sigma)
    else:
        raise ValueError('influence must be one of {"gaussian","square"}')

    density = float(np.sum(w))
    # sum_i (x_i - x) * w_i
    grad = (diffs * w[:, None]).sum(axis=0)

    return density, grad

def _hill_climb(x0: np.ndarray,
                data: np.ndarray,
                sigma: float,
                influence: str,
                step: float,
                tol: float,
                max_iter: int) -> Tuple[np.ndarray, float, int]:
    """
    Hill-climb from x0 along normalized gradient to reach a density attractor

    Parameters:
        x0: start point 
        data: dataset (
        sigma: influence parameter
        influence: "gaussian" | "square"
        step: delta step size 
        tol: tolerance for convergence 
        max_iter: maximum iterations

    Returns:
        x*: converged attractor position 
        f*: density at x*
        iters: number of iterations used
    """
    x = x0.astype(float).copy()
    for t in range(max_iter):
        fval, grad = _density_and_gradient_at(x, data, sigma, influence)
        norm_g = float(np.linalg.norm(grad))
        if norm_g < tol:
            return x, fval, t
        direction = grad / (norm_g + 1e-12)
        x_next = x + step * direction
        if np.linalg.norm(x_next - x) < tol:
            # converged in position
            fnext, _ = _density_and_gradient_at(x_next, data, sigma, influence)
            return x_next, fnext, t + 1
        x = x_next
    fval, _ = _density_and_gradient_at(x, data, sigma, influence)
    return x, fval, max_iter


def _merge_attractors(attractors: np.ndarray,
                      densities: np.ndarray,
                      merge_radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge nearby attractors using single-linkage style union

    Parameters:
        attractors: raw attractor positions
        densities: density at each attractor
        merge_radius: distance threshold for merging

    Returns:
        merged_centers: merged attractor centers (mean within each group)
        merged_dens: density proxy for each center (max density in group)
        memberships: index of merged center for each original attractor
    """
    m = attractors.shape[0]
    if m == 0:
        return attractors, densities, np.array([], dtype=int)

    # Union-Find
    parent = np.arange(m, dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Merge if distance <= merge_radius
    for i in range(m):
        for j in range(i + 1, m):
            if _euclidean_dist(attractors[i], attractors[j]) <= merge_radius:
                union(i, j)

    # Build groups
    root_to_indices: Dict[int, List[int]] = {}
    for i in range(m):
        r = find(i)
        root_to_indices.setdefault(r, []).append(i)

    k = len(root_to_indices)
    centers = np.zeros((k, attractors.shape[1]), dtype=float)
    dens_out = np.zeros(k, dtype=float)
    memberships = np.zeros(m, dtype=int)

    for new_idx, (_, idxs) in enumerate(root_to_indices.items()):
        centers[new_idx] = attractors[idxs].mean(axis=0)
        dens_out[new_idx] = float(np.max(densities[idxs]))
        for i in idxs:
            memberships[i] = new_idx

    return centers, dens_out, memberships

def _sample_line_density_ok(a: np.ndarray,
                            b: np.ndarray,
                            data: np.ndarray,
                            sigma: float,
                            influence: str,
                            xi: float,
                            step_len: float) -> bool:
    """
    Check whether the straight path from a to b can be traversed with density >= xi
    We sample points every 'step_len' distance and require all samples meet density >= xi
    """
    dist = np.linalg.norm(b - a)
    if dist < 1e-12:
        # same point
        f, _ = _density_and_gradient_at(a, data, sigma, influence)
        return f >= xi
    n_steps = int(np.ceil(dist / max(step_len, 1e-8)))
    for t in range(n_steps + 1):
        lam = t / max(n_steps, 1)
        p = (1 - lam) * a + lam * b
        f, _ = _density_and_gradient_at(p, data, sigma, influence)
        if f < xi:
            return False
    return True

def _connectivity_graph(centers: np.ndarray,
                        data: np.ndarray,
                        sigma: float,
                        influence: str,
                        xi: float,
                        step_len: float) -> List[List[int]]:
    """
    Build graph over attractor centers: connect i-j if there exists a straight path
    between them along which density >= xi (validated by sampling)

    Returns:
        adj: adjacency list as list of lists 
    """
    k = centers.shape[0]
    adj = [[] for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if _sample_line_density_ok(centers[i], centers[j], data, sigma, influence, xi, step_len):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    """Connected components on an undirected graph given as adjacency lists"""
    n = len(adj)
    vis = [False] * n
    comps = []
    for s in range(n):
        if vis[s]:
            continue
        stack = [s]
        vis[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


class DENCLUE():
    def __init__(self,
                 sigma: float = 1.0,
                 influence: str = "gaussian",
                 xi: Optional[float] = None,
                 step: float = 0.1,
                 tol: float = 1e-4,
                 max_iter: int = 200,
                 merge_radius: Optional[float] = None,
                 connectivity_step: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        """
        DENCLUE (DENsity-based CLUstEring)

        Parameters:
            sigma: Influence radius / bandwidth (>0)
            influence: "gaussian" or "square" influence function
            xi: Density threshold
            step: Hill-climbing step size and along normalized gradient
            tol: Convergence tolerance for hill-climbing
            max_iter: Maximum iterations in hill-climbing
            merge_radius: Radius to merge nearby density-attractors
            connectivity_step: Step length for sampling the straight path between two attractors when checking connectivity
            random_state: Optional random seed for reproducibility
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if influence not in {"gaussian", "square"}:
            raise ValueError('influence must be one of {"gaussian","square"}')
        if step <= 0:
            raise ValueError("step must be > 0")
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        self.sigma = float(sigma)
        self.influence = influence
        self.xi = xi
        self.step = float(step)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.merge_radius = 0.5 * self.sigma if merge_radius is None else float(merge_radius)
        self.connectivity_step = 0.5 * self.sigma if connectivity_step is None else float(connectivity_step)

        if random_state is not None:
            np.random.seed(int(random_state))

        # Learned attributes
        self.labels_ = None          
        self.attractors_ = None     
        self.attractor_densities_ = None  
        self.cluster_centers_ = None  

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit DENCLUE and return hard labels
        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels in {0..K-1}, noise = -1
        """
        X = np.asarray(features, dtype=float)
        n, d = X.shape
        if n == 0:
            self.labels_ = np.array([], dtype=int)
            self.attractors_ = np.zeros((0, d))
            self.attractor_densities_ = np.zeros((0,))
            self.cluster_centers_ = np.zeros((0, d))
            return self.labels_

        # Hill-climb for each data point
        raw_attractors = np.zeros_like(X)
        raw_densities = np.zeros(n, dtype=float)
        for i in range(n):
            a, f, _ = _hill_climb(X[i], X, self.sigma, self.influence, self.step, self.tol, self.max_iter)
            raw_attractors[i] = a
            raw_densities[i] = f

        # Merge nearby attractors
        merged_centers, merged_dens, memberships = _merge_attractors(
            raw_attractors, raw_densities, self.merge_radius
        )

        # Map each point to its merged attractor id
        point_attractor_id = memberships  

        # Choose xi if None
        if self.xi is None:
            # Heuristic: 60th percentile of merged attractor densities (robust, no tuning)
            xi = float(np.percentile(merged_dens, 60.0))
        else:
            xi = float(self.xi)

        # Keep only attractors with density >= xi
        keep_mask = merged_dens >= xi
        kept_idx = np.where(keep_mask)[0]
        if kept_idx.size == 0:
            # No significant attractor => all noise
            labels = -np.ones(n, dtype=int)
            self.labels_ = labels
            self.attractors_ = merged_centers
            self.attractor_densities_ = merged_dens
            self.cluster_centers_ = np.zeros((0, d))
            return labels

        # Relabel kept attractors to [0..k-1]
        old_to_new = {old: new for new, old in enumerate(kept_idx.tolist())}
        # For connectivity graph we only use kept centers
        kept_centers = merged_centers[kept_idx]
        kept_dens = merged_dens[kept_idx]

        # Connectivity among kept attractors (density along path >= xi)
        adj = _connectivity_graph(
            kept_centers, X, self.sigma, self.influence, xi, self.connectivity_step
        )
        comps = _connected_components(adj)  # list of lists of indices in kept_centers

        # Final clusters = connected components in attractor graph
        # Build final labels for points
        labels = -np.ones(n, dtype=int)
        # Map each kept attractor to its component label
        attractor_to_component: Dict[int, int] = {}
        for comp_label, comp_nodes in enumerate(comps):
            for a_idx in comp_nodes:
                attractor_to_component[a_idx] = comp_label

        for i in range(n):
            a_old = point_attractor_id[i]
            # If the merged attractor of point i is kept
            if keep_mask[a_old]:
                a_new = old_to_new[a_old]  # index in kept_centers
                labels[i] = attractor_to_component[a_new]
            else:
                labels[i] = -1  # noise

        # Save learned attributes
        self.labels_ = labels
        self.attractors_ = kept_centers
        self.attractor_densities_ = kept_dens
        # Cluster centers = mean of attractor centers per component (not strictly required)
        K = len(comps)
        cluster_centers = np.zeros((K, d), dtype=float)
        for k, comp_nodes in enumerate(comps):
            cluster_centers[k] = kept_centers[np.array(comp_nodes)].mean(axis=0)
        self.cluster_centers_ = cluster_centers

        return labels

    def __str__(self) -> str:
        return "DENCLUE (Density-based Clustering via Influence Functions)"
