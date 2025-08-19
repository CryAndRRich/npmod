from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import heapq

def pairwise_distances(features: np.ndarray) -> np.ndarray:
    """Compute full Euclidean pairwise distance matrix"""
    sq = np.sum(features * features, axis=1, keepdims=True)  
    D2 = sq + sq.T - 2.0 * (features @ features.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, out=D2)

def core_distances(D: np.ndarray, 
                   min_samples: int) -> np.ndarray:
    """
    Core distance of each point = distance to the min_samples-th nearest neighbor

    Parameters:
        D: Pairwise distance matrix 
        min_samples: k for k-distance
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    
    # sort each row; index 0 is self (0), so take [min_samples]
    sorted_rows = np.sort(D, axis=1)
    k = min(min_samples, D.shape[0] - 1)

    core = sorted_rows[:, k] if k > 0 else np.zeros(D.shape[0], dtype=float)
    return core

def mutual_reachability(D: np.ndarray, 
                        core: np.ndarray) -> np.ndarray:
    """
    Mutual reachability distance matrix:
        mr(a,b) = max( core[a], core[b], D[a,b] )

    Parameters:
        D: Pairwise distances
        core: Core distances
    """
    ca = core[:, None]
    cb = core[None, :]
    mr = np.maximum(np.maximum(ca, cb), D)
    np.fill_diagonal(mr, 0.0)
    return mr

def mst_prim(W: np.ndarray) -> List[Tuple[float, int, int]]:
    """
    Minimum Spanning Tree (Prim) on dense symmetric weight matrix

    Parameters:
        W: symmetric weights (mutual reachability)

    Returns:
        edges: list of (weight, u, v)
    """
    n = W.shape[0]
    visited = [False] * n
    edges = []
    pq = []  # (w, v, parent)

    # start from node 0
    heapq.heappush(pq, (0.0, 0, -1))
    while pq:
        w, v, p = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        if p != -1:
            edges.append((w, p, v))
        for u in range(n):
            if not visited[u] and u != v:
                heapq.heappush(pq, (W[v, u], u, v))
    return edges

class ClusterNode():
    """
    Internal node representing a cluster in the condensed tree.
    """
    __slots__ = ("id", "points", "lambda_birth", "lambda_death",
                 "stability", "children", "parent")

    def __init__(self,
                 cid: int,
                 points: Set[int],
                 lambda_birth: float,
                 parent: Optional[int]) -> None:
        
        self.id = cid
        self.points = set(points)          
        self.lambda_birth = float(lambda_birth)
        self.lambda_death = float("inf")    
        self.stability = 0.0                
        self.children = []           
        self.parent = parent


class HDBSCAN():
    def __init__(self,
                 min_samples: int = 5,
                 min_cluster_size: Optional[int] = None) -> None:
        """
        HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise

        Parameters:
            min_samples: Defines local density (k for k-distance)
            min_cluster_size: Minimum size to keep a cluster in condensed tree. If None, defaults to min_samples
        """
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        
        self.min_samples = int(min_samples)
        self.min_cluster_size = int(min_cluster_size) if min_cluster_size is not None else int(min_samples)

        self.labels_ = None

        self.clusters_ = {}
        self.selected_clusters_ = []

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN and return hard labels (-1 for noise)

        Parameters:
            features: Feature matrix of the training data 

        Returns:
            labels: Cluster labels for each sample (-1 indicates noise)
        """
        n = features.shape[0]
        if n == 0:
            self.labels_ = np.array([], dtype=int)
            return self.labels_

        # Distances & core distances
        D = pairwise_distances(features)
        core = core_distances(D, self.min_samples)

        # Mutual reachability & MST
        mr = mutual_reachability(D, core)
        mst_edges = mst_prim(mr)  # [(w,u,v)]
        if len(mst_edges) == 0:
            # All points isolated => all noise
            self.labels_ = -np.ones(n, dtype=int)
            return self.labels_

        # Build hierarchy by removing heaviest edges first
        labels = self._build_hierarchy_and_select(mst_edges, n)

        self.labels_ = labels
        return labels

    def _build_hierarchy_and_select(self,
                                    mst_edges: List[Tuple[float, int, int]],
                                    n: int) -> np.ndarray:
        """
        Build hierarchy from MST, condense it (min_cluster_size),
        compute cluster stability, and select optimal flat clustering
        """
        # Build adjacency from MST
        adj = {i: set() for i in range(n)}
        for w, u, v in mst_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Sort MST edges by descending weight 
        edges_desc = sorted(mst_edges, key=lambda x: x[0], reverse=True)

        # Track current set of "active" clusters (by id -> node)
        next_id = 0
        clusters = {}
        # Root cluster: all points live from λ=1/max_w
        max_w = max(w for w, _, _ in mst_edges)
        lambda_start = 0.0 if max_w <= 0 else 1.0 / (max_w + 1e-12)
        root = ClusterNode(next_id, set(range(n)), lambda_start, parent=None)
        clusters[next_id] = root
        active = {next_id: root}
        next_id += 1

        # Helper to get connected component after temporarily removing one edge, restricted to a node set
        def split_component_by_edge(node_points: Set[int], 
                                    a: int, 
                                    b: int) -> Tuple[Set[int], Set[int]]:
            """
            Remove edge (a,b) from adjacency and compute the two connected components
            restricted to node_points. Assumes that within node_points, the edge
            (a,b) is a bridge in the MST, hence exactly 2 components
            """
            # Do two DFS starting from a and b, not crossing the removed edge
            compA = set()
            stack = [a]
            visited_local = set()
            while stack:
                u = stack.pop()
                if u in visited_local or u not in node_points:
                    continue
                visited_local.add(u)
                compA.add(u)
                for w in adj[u]:
                    if (u == a and w == b) or (u == b and w == a):
                        continue
                    if w not in visited_local:
                        stack.append(w)
            compB = node_points - compA
            return compA, compB

        # Process edges from heaviest to lightest 
        for w, u, v in edges_desc:
            lam = float("inf") if w <= 0 else 1.0 / (w + 1e-12)

            # Find the active cluster containing both endpoints u and v
            container_id = None
            for cid, node in active.items():
                if u in node.points and v in node.points:
                    container_id = cid
                    break
            if container_id is None:
                # endpoints already separated in earlier splits
                # remove edge in adjacency and continue
                adj[u].discard(v)
                adj[v].discard(u)
                continue

            container = active[container_id]

            # Remove this edge from adjacency and split the container
            adj[u].discard(v)
            adj[v].discard(u)
            A, B = split_component_by_edge(container.points, u, v)
            if not A or not B:
                # no actual split (shouldn't happen with MST), continue
                continue

            sizeA, sizeB = len(A), len(B)
            mcs = self.min_cluster_size

            # Three cases for condensed tree (HDBSCAN condensing logic):
            if sizeA >= mcs and sizeB >= mcs:
                # Both are "large": container dies at lam, two large children are born
                # All container's remaining points leave *now*: add to stability
                leaving = len(container.points)
                container.stability += leaving * (lam - container.lambda_birth)
                container.lambda_death = lam

                # Create children 
                childA = ClusterNode(next_id, A, lam, parent=container.id)
                next_id += 1
                childB = ClusterNode(next_id, B, lam, parent=container.id)
                next_id += 1
                container.children.extend([childA.id, childB.id])

                # Replace in active set
                del active[container.id]
                active[childA.id] = childA
                active[childB.id] = childB
                clusters[childA.id] = childA
                clusters[childB.id] = childB

            elif sizeA >= mcs and sizeB < mcs:
                # A is a large child; B is small and becomes noise relative to this branch
                # The B points leave at lam, contributing to container's stability
                container.stability += len(B) * (lam - container.lambda_birth)

                # Shrink container to the remaining points
                container.points -= B

                # Create the large child cluster for A
                childA = ClusterNode(next_id, A, lam, parent=container.id)
                next_id += 1
                container.children.append(childA.id)
                active[childA.id] = childA
                clusters[childA.id] = childA

            elif sizeA < mcs and sizeB >= mcs:
                # Symmetric to above
                container.stability += len(A) * (lam - container.lambda_birth)
                container.points -= A

                childB = ClusterNode(next_id, B, lam, parent=container.id)
                next_id += 1
                container.children.append(childB.id)
                active[childB.id] = childB
                clusters[childB.id] = childB

            else:
                # Both children small: entire container dissolves into noise at lam
                container.stability += len(container.points) * (lam - container.lambda_birth)
                container.lambda_death = lam
                del active[container.id]
                # no children recorded

        lambda_end = max((1.0 / (w + 1e-12) if w > 0 else float("inf")) for w, _, _ in mst_edges)
        for node in list(active.values()):
            node.stability += len(node.points) * (lambda_end - node.lambda_birth)
            node.lambda_death = lambda_end

        # Save clusters for optional inspection/plot
        self.clusters_ = clusters

        # Optimal cluster selection (parent-vs-children by stability)
        chosen = self._select_optimal_clusters(clusters)
        self.selected_clusters_ = chosen

        # Hard labeling: points in chosen clusters get cluster ids; others = -1 (noise)
        labels = -np.zeros(n, dtype=int) - 1  # init to -1

        # Assign by chosen clusters; if overlaps (shouldn't), prefer smaller (deeper) cluster
        # So sort chosen by decreasing lambda_birth (deeper clusters have larger birth λ)
        chosen_sorted = sorted(chosen, key=lambda cid: clusters[cid].lambda_birth, reverse=True)
        cid_to_label: Dict[int, int] = {}
        label_counter = 0
        for cid in chosen_sorted:
            cid_to_label[cid] = label_counter
            pts = clusters[cid].points
            labels[list(pts)] = label_counter
            label_counter += 1

        return labels

    def _select_optimal_clusters(self, clusters: Dict[int, ClusterNode]) -> List[int]:
        """Non-overlapping selection of clusters maximizing total stability"""
        # Build tree structure
        children_map = {cid: list(node.children) for cid, node in clusters.items()}
        parent_map = {cid: node.parent for cid, node in clusters.items()}

        # Find roots (usually 1 root)
        roots = [cid for cid, p in parent_map.items() if p is None]

        chosen = set()

        def dp_select(cid: int) -> Tuple[float, Set[int]]:
            node = clusters[cid]
            if not children_map[cid]:
                return node.stability, {cid}
            # sum of optimal children selections
            sum_child = 0.0
            chosen_child: Set[int] = set()
            for ch in children_map[cid]:
                score, chosen_set = dp_select(ch)
                sum_child += score
                chosen_child |= chosen_set
            # compare with parent stability
            if node.stability >= sum_child:
                return node.stability, {cid}
            else:
                return sum_child, chosen_child

        for r in roots:
            _, chosen_set = dp_select(r)
            chosen |= chosen_set

        return list(chosen)
    
    def __str__(self) -> str:
        return "HDBSCAN: (Hierarchical DBSCAN)"
