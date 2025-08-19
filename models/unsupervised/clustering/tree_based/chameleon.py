import numpy as np
import heapq
from typing import Dict, List, Optional, Set, Tuple

class CHAMELEON():
    def __init__(self,
                 number_of_clusters: int = 2,
                 n_neighbors: int = 10,
                 alpha: float = 0.5,
                 mutual_knn: bool = True,
                 similarity: str = "gaussian",
                 gaussian_sigma: Optional[float] = None,
                 min_component_size: int = 1,
                 random_state: Optional[int] = None) -> None:
        """
        Parameters:
            number_of_clusters: Target number of clusters to output (K_out)
            n_neighbors: Number of neighbors for the kNN graph
            alpha: Trade-off between RC and RCL in the merge score
            mutual_knn: If True, keep an undirected edge (i, j) only when i is in j's neighbors AND j is in i's neighbors (mutual kNN graph). If False, we symmetrize the directed kNN graph
            similarity: "gaussian" or "inverse_distance"
            gaussian_sigma: Sigma for the Gaussian similarity. If None, will use the median of neighbor distances
            min_component_size: After constructing the kNN graph, discard connected components whose size < min_component_size
            random_state: Random state is not strictly needed here
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if similarity not in {"gaussian", "inverse_distance"}:
            raise ValueError('similarity must be one of {"gaussian", "inverse_distance"}')
        if min_component_size < 1:
            raise ValueError("min_component_size must be >= 1")

        self.n_clusters = number_of_clusters
        self.k = n_neighbors
        self.alpha = alpha
        self.mutual_knn = mutual_knn
        self.similarity = similarity
        self.gaussian_sigma = gaussian_sigma
        self.min_component_size = min_component_size
        self.random_state = random_state

        self.labels_ = None
        self.centroids_ = None

    @staticmethod
    def _pairwise_distances(features: np.ndarray) -> np.ndarray:
        """Compute dense Euclidean pairwise distance matrix"""
        sq = np.sum(features * features, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (features @ features.T)
        np.maximum(D2, 0.0, out=D2)
        return np.sqrt(D2, out=D2)

    def _build_knn_graph(self, features: np.ndarray) -> Tuple[Dict[int, Dict[int, float]], float]:
        """
        Build (mutual) kNN graph with chosen similarity

        Returns:
            adj: Adjacency dictionary
            sigma: The sigma used if Gaussian similarity is selected
        """
        n = features.shape[0]
        k = min(self.k, n - 1)
        if k <= 0:
            return {i: {} for i in range(n)}, 1.0

        D = self._pairwise_distances(features)

        # For each i, get indices of k nearest neighbors (exclude i)
        nn_idx = np.argsort(D, axis=1)[:, 1: k + 1]
        nn_dist = np.take_along_axis(D, nn_idx, axis=1)

        # Set sigma if needed
        if self.similarity == "gaussian":
            if self.gaussian_sigma is not None and self.gaussian_sigma > 0:
                sigma = float(self.gaussian_sigma)
            else:
                # Use median of neighbor distances as scale
                sigma = float(np.median(nn_dist))
                if sigma <= 0:
                    sigma = 1.0
        else:
            sigma = 1.0  # not used

        def sim_from_dist(dist: float) -> float:
            if self.similarity == "gaussian":
                # exp(-d^2 / (2 sigma^2))
                return float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))
            else:
                # 1 / (1 + d)
                return float(1.0 / (1.0 + dist))

        # Build directed neighbor sets
        out_nbrs = [set(nn_idx[i]) for i in range(n)]
        adj = {i: {} for i in range(n)}

        # Add edges
        for i in range(n):
            for rank, j in enumerate(nn_idx[i]):
                if self.mutual_knn and (i not in out_nbrs[j]):
                    continue  # keep only mutual edges
                # add undirected edge (i, j) once: store into the smaller key to avoid duplicates
                a, b = (i, j) if i < j else (j, i)
                if b not in adj[a]:
                    w = sim_from_dist(D[i, j])
                    adj[a][b] = w

        return adj, sigma

    @staticmethod
    def _connected_components(n: int, 
                              adj: Dict[int, Dict[int, float]]) -> List[List[int]]:
        """
        Find connected components in an undirected graph represented by half adjacency (a<b)
        """
        visited = [False] * n
        neighbors = [set() for _ in range(n)]
        for a, nbrs in adj.items():
            for b in nbrs:
                neighbors[a].add(b)
                neighbors[b].add(a)

        comps = []
        for s in range(n):
            if visited[s]:
                continue
            # BFS
            q = [s]
            visited[s] = True
            comp = []
            while q:
                u = q.pop()
                comp.append(u)
                for v in neighbors[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            comps.append(comp)
        return comps

    @staticmethod
    def _cluster_internal_edge_stats(nodes: Set[int], 
                                     adj: Dict[int, Dict[int, float]]) -> Tuple[float, int]:
        """
        Sum of internal edge weights and number of internal edges for a cluster.
        Graph is stored as half adjacency (a<b). Count each undirected edge once
        """
        nodes_sorted = sorted(nodes)
        node_set = set(nodes_sorted)
        w_sum = 0.0
        m = 0
        for a in nodes_sorted:
            nbrs = adj.get(a, {})
            for b, w in nbrs.items():
                if b in node_set:
                    w_sum += w
                    m += 1
        return w_sum, m

    @staticmethod
    def _between_edge_stats(A: Set[int], 
                            B: Set[int], 
                            adj: Dict[int, Dict[int, float]]) -> Tuple[float, int]:
        """
        Sum of edge weights and edge count between clusters A and B
        """
        if len(A) == 0 or len(B) == 0:
            return 0.0, 0
        if len(A) > len(B):
            A, B = B, A  # iterate the smaller for efficiency
        B_set = set(B)
        w_sum = 0.0
        m = 0
        for a in A:
            # edges stored as adj[min][max]
            for b, w in adj.get(min(a, max(B_set)), {}).items():
                # This shortcut is not general; fallback to safe traversal below.
                pass  # we'll do a safe approach instead

        # Safe approach: traverse all half-edges and count if endpoints split across A and B
        for a, nbrs in adj.items():
            for b, w in nbrs.items():
                in_a = (a in A) + (b in A)
                in_b = (a in B) + (b in B)
                if in_a == 1 and in_b == 1:  # exactly one endpoint in each
                    w_sum += w
                    m += 1
        return w_sum, m

    @staticmethod
    def _safe_div(num: float, 
                  den: float, 
                  eps: float = 1e-12) -> float:
        return num / (den + eps)

    def _merge_score(self,
                     A: Set[int],
                     B: Set[int],
                     adj: Dict[int, Dict[int, float]],
                     cache_int: Dict[int, Tuple[float, int]],
                     idA: int,
                     idB: int) -> float:
        """
        Compute CHAMELEON merge score for clusters A and B using RC and RCL
        """
        # Internal stats (cached)
        EC_A, mA = cache_int[idA]
        EC_B, mB = cache_int[idB]

        # Between stats
        EC_AB, mAB = self._between_edge_stats(A, B, adj)

        # RC = EC(A,B) / ((EC_int(A)+EC_int(B))/2)
        RC = self._safe_div(EC_AB, 0.5 * (EC_A + EC_B))

        # RCL = avg_between / ((avg_int(A)+avg_int(B))/2)
        avg_between = self._safe_div(EC_AB, float(mAB))
        avg_int_A = self._safe_div(EC_A, float(mA))
        avg_int_B = self._safe_div(EC_B, float(mB))
        RCL = self._safe_div(avg_between, 0.5 * (avg_int_A + avg_int_B))

        # Weighted product
        score = (RC ** self.alpha) * (RCL ** (1.0 - self.alpha))
        return float(score)

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Run CHAMELEON clustering on data

        Parameters:
            features: Input feature matrix

        Returns:
            labels: Cluster assignment (0..K-1)
        """
        n = features.shape[0]
        if self.n_clusters > n:
            raise ValueError("number_of_clusters cannot exceed number of samples")

        # kNN graph
        adj, _ = self._build_knn_graph(features)

        # Initial partitions: connected components (seed clusters)
        comps = self._connected_components(n, adj)
        comps = [c for c in comps if len(c) >= self.min_component_size]
        if len(comps) == 0:
            # If graph is too sparse, fall back to singletons
            comps = [[i] for i in range(n)]

        # If fewer components than requested clusters, pad with singletons (rare)
        if len(comps) < self.n_clusters:
            present = set(np.concatenate([np.array(c) for c in comps])) if comps else set()
            missing = [i for i in range(n) if i not in present]
            for i in missing:
                comps.append([i])
                if len(comps) >= self.n_clusters:
                    break

        # Prepare cluster bookkeeping
        clusters = {i: set(comp) for i, comp in enumerate(comps)}
        active = set(clusters.keys())
        next_id = len(clusters)

        # Precompute internal stats cache: cluster_id -> (EC_int, m_int)
        cache_int = {}
        for cid, nodes in clusters.items():
            EC, m = self._cluster_internal_edge_stats(nodes, adj)
            cache_int[cid] = (EC, m)

        # Build initial candidate pairs with scores (max-heap via negative score)
        heap = []
        ids = list(active)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                score = self._merge_score(clusters[a], clusters[b], adj, cache_int, a, b)
                if score > 0.0:
                    heapq.heappush(heap, (-score, a, b))

        # Agglomerative merging
        while len(active) > self.n_clusters:
            # Pop best available pair
            found = False
            while heap:
                _, a, b = heapq.heappop(heap)
                if a in active and b in active:
                    found = True
                    break
            if not found:
                # No more mergeable pairs; break early
                break

            # Merge clusters a and b into new cluster 'next_id'
            A = clusters[a]
            B = clusters[b]
            new_nodes = A | B

            # Update internal stats:
            EC_A, mA = cache_int[a]
            EC_B, mB = cache_int[b]
            EC_AB, mAB = self._between_edge_stats(A, B, adj)

            EC_new = EC_A + EC_B + EC_AB         # internal edges are union plus cross edges become internal
            m_new = mA + mB + mAB

            # Commit merge
            clusters[next_id] = new_nodes
            cache_int[next_id] = (EC_new, m_new)

            # Remove old
            del clusters[a], clusters[b]
            del cache_int[a], cache_int[b]
            active.remove(a)
            active.remove(b)
            active.add(next_id)

            # Add new candidate pairs with scores
            for w in list(active):
                if w == next_id:
                    continue
                score = self._merge_score(clusters[next_id], clusters[w], adj, cache_int, next_id, w)
                if score > 0.0:
                    heapq.heappush(heap, (-score, next_id, w))

            next_id += 1

        # Map remaining active cluster ids to [0..K-1]
        id_to_label = {cid: k for k, cid in enumerate(active)}
        labels = np.empty(n, dtype=int)
        for cid, nodes in clusters.items():
            if cid in id_to_label:
                labels[list(nodes)] = id_to_label[cid]

        # Centroids
        K = len(active)
        centroids = np.zeros((K, features.shape[1]), dtype=float)
        for cid, k in id_to_label.items():
            idx = np.fromiter(clusters[cid], dtype=int)
            centroids[k] = features[idx].mean(axis=0) if idx.size > 0 else np.zeros(features.shape[1], dtype=float)

        self.labels_ = labels
        self.centroids_ = centroids
        return labels

    def __str__(self) -> str:
        return "CHAMELEON (Graph-based Agglomerative Clustering)"
