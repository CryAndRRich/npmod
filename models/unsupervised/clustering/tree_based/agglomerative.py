import heapq
import numpy as np

class AgglomerativeClustering():
    def __init__(self,
                 number_of_clusters: int = 2,
                 linkage: str = "single") -> None:
        """
        Agglomerative (hierarchical) clustering using specified linkage criteria

        Parameters:
            number_of_clusters: The number of clusters to form
            linkage: Linkage criterion to use - one of "single", "complete", "avg", "centroid"
        """
        self.n_clusters = number_of_clusters
        self.linkage = linkage.lower()
        if self.linkage not in {"single", "complete", "average", "centroid", "ward"}:
            raise ValueError(f"Unsupported linkage: {linkage}")
        self.labels_ = None
        self.children_ = None
        self.distances_ = None
        self.counts_ = None

    def _pairwise_distance(self, features: np.ndarray) -> np.ndarray:
        """Compute full pairwise Euclidean distance matrix"""
        diff = features[:, None, :] - features[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
    
    def _lw_update(self, 
                   link: str, 
                   na: int, 
                   nb: int, 
                   nw: int,
                   dab: float, 
                   daw: float, 
                   dbw: float) -> float:
        
        if link == "single":
            return min(daw, dbw)
        if link == "complete":
            return max(daw, dbw)
        if link == "average": 
            return (na * daw + nb * dbw) / (na + nb)
        if link == "centroid":
            na_nb = na + nb
            d2 = (na / na_nb) * (daw ** 2) + (nb / na_nb) * (dbw ** 2) - (na * nb / (na_nb ** 2)) * (dab ** 2)
            return float(np.sqrt(max(d2, 0.0)))
        if link == "ward":
            denom = na + nb + nw
            d2 = ((na + nw) * (daw ** 2) + (nb + nw) * (dbw ** 2) - nw * (dab ** 2)) / denom
            return float(np.sqrt(max(d2, 0.0)))
        raise RuntimeError("Unexpected linkage")
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Performs hierarchical agglomerative clustering on the data

        Parameters:
            features: Feature matrix of the training data

        Returns:
            labels: Cluster labels for each sample (0 to k-1)
        """
        n = features.shape[0]
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters must be in [1, n_samples]")

        # initial distances among leaves 0..n-1
        D = self._pairwise_distance(features)

        # active cluster sizes and a dict for pair distances (current)
        size = {i: 1 for i in range(n)}
        # store distances in dict using sorted pair keys for quick access
        pair_dist = {}
        heap = []

        for i in range(n):
            for j in range(i+1, n):
                d = D[i, j]
                pair_dist[(i, j)] = d
                # heap item: (distance, a, b) with a<b for stable tie-breaking
                heapq.heappush(heap, (d, i, j))

        active = set(range(n))
        next_id = n  # new internal node ids: n, n+1, ...
        merges_needed = n - 1

        children = np.empty((merges_needed, 2), dtype=int)
        distances = np.empty((merges_needed,), dtype=float)
        counts = np.empty((merges_needed,), dtype=int)

        # For cutting later, we need to know children of each internal node
        node_children = {}

        k_merge = 0
        while len(active) > 1:
            # pop nearest valid pair
            while True:
                d, a, b = heapq.heappop(heap)
                if a > b:
                    a, b = b, a
                if (a in active) and (b in active):
                    # ensure distance is still current (may be stale)
                    cur = pair_dist.get((a, b), None)
                    if cur is not None and abs(cur - d) <= 1e-12:
                        break

            # record merge
            children[k_merge] = [a, b]
            distances[k_merge] = d
            new_id = next_id
            next_id += 1
            new_size = size[a] + size[b]
            counts[k_merge] = new_size
            node_children[new_id] = (a, b)

            # deactivate a,b; activate new_id
            active.remove(a); active.remove(b)
            active.add(new_id)

            # remove obsolete pair distances touching a or b
            size[new_id] = new_size

            # update distances from new_id to every other active w
            for w in list(active):
                if w == new_id:
                    continue
                aa, bb = (a, w) if a < w else (w, a)
                ba, bb2 = (b, w) if b < w else (w, b)
                daw = pair_dist.get((aa, bb), np.inf)
                dbw = pair_dist.get((ba, bb2), np.inf)
                dab = d  # current distance between a and b

                # sizes
                na, nb, nw = size[a], size[b], size[w]
                new_dw = self._lw_update(self.linkage, na, nb, nw, dab, daw, dbw)

                key = (w, new_id) if w < new_id else (new_id, w)
                pair_dist[key] = new_dw
                heapq.heappush(heap, (new_dw, key[0], key[1]))

            # clean sizes of merged leaves to avoid accidental use
            del size[a]; del size[b]

            k_merge += 1
            if len(active) == self.n_clusters:
                pass

        # save tree arrays
        self.children_ = children
        self.distances_ = distances
        self.counts_ = counts

        # final root id = next_id-1
        root = next_id - 1
        # components = list of node ids (some internal >= n, or leaf < n)
        components = [root]

        # helper: get distance height of a node (0 for leaf)
        def node_height(node_id: int) -> float:
            if node_id < n:
                return 0.0
            # node_id = n + idx  => idx = node_id - n
            return self.distances_[node_id - n]

        # split the largest-height node until we have the desired number of components
        while len(components) < self.n_clusters:
            # pick the component with the largest height (tie -> largest id to be deterministic)
            pick_idx = max(range(len(components)),
                           key=lambda i: (node_height(components[i]), components[i]))
            node = components.pop(pick_idx)
            if node < n:
                components.insert(pick_idx, node)
                break
            left, right = node_children[node]
            components.extend([left, right])

        # assign labels: DFS all leaves under each component; order by components as-is
        labels = np.empty(n, dtype=int)

        def assign(node_id: int, label: int):
            if node_id < n:
                labels[node_id] = label
            else:
                l, r = node_children[node_id]
                assign(l, label)
                assign(r, label)

        for lab, comp in enumerate(components):
            assign(comp, lab)

        self.labels_ = labels
        return labels

    def __str__(self) -> str:
        return "Agglomerative Clustering"
