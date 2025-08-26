from typing import Tuple
import numpy as np

class UMAP():
    def __init__(self, 
                 n_neighbors: int = 15,
                 min_dist: float = 0.1,
                 n_components: int = 2,
                 n_epochs: int = 200,
                 learning_rate: float = 1.0,
                 negative_sample_rate: int = 5,
                 random_state: int = 42) -> None:
        """
        Initialize UMAP reducer with hyperparameters

        Parameters:
            n_neighbors: Number of nearest neighbors to use for local approximations
            min_dist: Minimum distance between points in low-dimensional space
            n_components: Target dimension of the embedding
            n_epochs: Number of optimization epochs
            learning_rate: Step size for embedding optimization
            negative_sample_rate: Number of negative samples per positive edge in each epoch
            random_state: Seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state

        # Fitted attributes
        self.embedding_ = None         
        self._X_fit = None           
        self._edges = []   
        self._knn_idx = None        
        self._knn_dist = None       

        # Low-D curve parameters (standard UMAP constants)
        self._a = 1.929
        self._b = 0.7915

    @staticmethod
    def _pairwise_distance(X: np.ndarray) -> np.ndarray:
        sum_X = np.sum(np.square(X), axis=1)
        D = -2.0 * (X @ X.T) + sum_X[:, None] + sum_X[None, :]
        np.fill_diagonal(D, 0.0)
        return D

    @staticmethod
    def _row_knn(dist_sq: np.ndarray, 
                 k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return indices and squared distances to k nearest neighbors for each row"""

        # argsort full (O(n log n) per row) – fine for small n
        idx = np.argsort(dist_sq, axis=1)[:, 1:k+1]
        dists = np.take_along_axis(dist_sq, idx, axis=1)
        return idx, dists

    def _smooth_knn_dist(self, 
                         knn_dists: np.ndarray, 
                         tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-point by binary search"""

        n, k = knn_dists.shape
        rho = knn_dists[:, 0].copy()   # distance to nearest neighbor
        sigma = np.zeros(n, dtype=np.float64)
        target = np.log2(k) if k > 1 else 1.0

        for i in range(n):
            lo, hi = 0.0, np.inf
            mid = 1.0
            di = knn_dists[i]

            # Handle edge cases if neighbors all zero-dist (duplicates)
            if np.allclose(di, di[0]):
                sigma[i] = 1.0
                continue

            for _ in range(64):
                x = -(di - rho[i]) / (mid + 1e-12)
                x = np.clip(x, -50.0, 50.0)  # numeric stability
                ps = np.exp(x)
                ps[di < rho[i]] = 1.0  # distances below rho contribute fully
                s = ps.sum()

                if abs(s - target) < tol:
                    break
                if s > target:
                    lo = mid
                    mid = mid * 2 if np.isinf(hi) else (mid + hi) / 2
                else:
                    hi = mid
                    mid = mid / 2 if lo == 0.0 else (mid + lo) / 2
            sigma[i] = mid

        return rho, sigma

    @staticmethod
    def _fuzzy_union(Pi: np.ndarray, Pj: np.ndarray) -> np.ndarray:
        return Pi + Pj - Pi * Pj

    def _build_graph(self, X: np.ndarray) -> None:
        """Build fuzzy simplicial set (as an edge list) using kNN and smooth kNN distances"""

        n = X.shape[0]
        D = self._pairwise_distance(X)
        knn_idx, knn_dist = self._row_knn(D, self.n_neighbors)
        self._knn_idx, self._knn_dist = knn_idx, knn_dist

        rho, sigma = self._smooth_knn_dist(knn_dist)

        # Row-stochastic local memberships
        P_row = np.zeros_like(knn_dist, dtype=np.float64)
        for i in range(n):
            # P_i,j over neighbors j of i
            numer = np.exp(-(knn_dist[i] - rho[i]) / (sigma[i] + 1e-12))
            numer[knn_dist[i] < rho[i]] = 1.0  # below rho -> full membership
            # No explicit row-normalization here (UMAP uses fuzzy set logic)
            P_row[i] = np.clip(numer, 0.0, 1.0)

        # Fuzzy union to symmetrize
        edges = {}
        for i in range(n):
            for col, j in enumerate(knn_idx[i]):
                w_ij = P_row[i, col]
                if w_ij <= 0.0:
                    continue
                if i < j:
                    key = (i, j)
                else:
                    key = (j, i)

                if key not in edges:
                    # try find reverse weight
                    w_ji = 0.0
                    js = knn_idx[j]
                    pos = np.where(js == i)[0]
                    if pos.size > 0:
                        w_ji = P_row[j, pos[0]]
                    w = w_ij + w_ji - w_ij * w_ji
                    if w > 0:
                        edges[key] = w
                else:
                    # combine again if encountered twice
                    w_prev = edges[key]
                    w = w_prev + w_ij - w_prev * w_ij
                    edges[key] = w

        # Normalize weights so that mean edge weight ~ 1.0 (optional, keeps scales reasonable)
        if edges:
            w_mean = np.mean(list(edges.values()))
            if w_mean > 0:
                for k in list(edges.keys()):
                    edges[k] = edges[k] / w_mean

        # store edges list
        self._edges = [(i, j, float(w)) for (i, j), w in edges.items() if w > 0.0]

    def _attractive_grad(self, diff: np.ndarray, a: float, b: float) -> np.ndarray:
        dist2 = np.sum(diff * diff, axis=1) + 1e-12
        dist_b = np.power(dist2, b)             
        denom = (1.0 + a * dist_b)
        w = 1.0 / denom                        
       
        coef = (2.0 * a * b) * np.power(dist2, b - 1.0) * w
        return (coef[:, None]) * diff 

    def _repulsive_grad(self, diff: np.ndarray) -> np.ndarray:
        dist2 = np.sum(diff * diff, axis=1) + 1e-12
        w = 1.0 / (1.0 + dist2)
        coef = 2.0 * w  # strength
        return (coef[:, None]) * diff

    def fit(self, features: np.ndarray) -> None:
        """
        Fit UMAP to the input data and compute low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            self: Fitted UMAP instance with embedding_ attribute
        """
        rng = np.random.RandomState(self.random_state)
        X = features.astype(np.float64, copy=False)
        n = X.shape[0]

        self._X_fit = X
        self._build_graph(X)  # sets self._edges

        # Init embedding (small random)
        Y = rng.normal(scale=1e-4, size=(n, self.n_components))

        # Pre-build neighbor sets for negative sampling
        neighbors = [set() for _ in range(n)]
        for i, j, _ in self._edges:
            neighbors[i].add(j)
            neighbors[j].add(i)

        # Optimization
        for epoch in range(1, self.n_epochs + 1):
            grad = np.zeros_like(Y)

            # Process in small batches for cache locality
            batch_size = max(1, len(self._edges) // 10)
            for start in range(0, len(self._edges), batch_size):
                subset = self._edges[start:start+batch_size]
                if not subset:
                    continue
                I = np.fromiter((e[0] for e in subset), dtype=np.int64)
                J = np.fromiter((e[1] for e in subset), dtype=np.int64)
                W = np.fromiter((e[2] for e in subset), dtype=np.float64)

                diff = Y[I] - Y[J] 
                g = self._attractive_grad(diff, self._a, self._b)
                g *= W[:, None]

                np.add.at(grad, I, g)
                np.add.at(grad, J, -g)

                if self.negative_sample_rate > 0:
                    # For each positive (i,j), sample ns negatives for i
                    ns = self.negative_sample_rate * len(subset)
                    i_neg = rng.randint(0, n, size=ns)
                    j_neg = rng.randint(0, n, size=ns)
                    # avoid sampling neighbors, resample a few times
                    mask_bad = (i_neg == j_neg)
                    for _ in range(3):
                        # also try to avoid true neighbors
                        mask_bad |= np.fromiter((j_neg[t] in neighbors[i_neg[t]] for t in range(ns)),
                                                dtype=bool, count=ns)
                        if not mask_bad.any():
                            break
                        repl = rng.randint(0, n, size=mask_bad.sum())
                        i_neg[mask_bad] = repl
                        j_neg[mask_bad] = rng.randint(0, n, size=mask_bad.sum())
                        mask_bad = (i_neg == j_neg)

                    diff_neg = Y[i_neg] - Y[j_neg]
                    g_rep = self._repulsive_grad(diff_neg)  
                    # scale repulsive term modestly
                    g_rep *= 0.25
                    np.add.at(grad, i_neg, g_rep)
                    np.add.at(grad, j_neg, -g_rep)

            # Gradient descent step: Y <- Y - lr * grad
            Y -= self.learning_rate * grad

            # zero-mean to avoid drift
            Y -= Y.mean(axis=0, keepdims=True)

        self.embedding_ = Y

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform new data into existing UMAP embedding space via weighted average of neighbors

        Parameters:
            features: New data matrix 

        Returns:
            embedding: Transformed data
        """
        if self.embedding_ is None or self._X_fit is None or self._knn_idx is None:
            raise ValueError("UMAP model has not been fitted yet. Call fit() first.")

        X_new = features.astype(np.float64, copy=False)
        Y_train = self.embedding_
        X_train = self._X_fit

        # Compute distances 
        sum_new = np.sum(X_new * X_new, axis=1)[:, None]
        sum_train = np.sum(X_train * X_train, axis=1)[None, :]
        D = -2.0 * (X_new @ X_train.T) + sum_new + sum_train

        # For each new point, pick the same k used in fit
        k = self.n_neighbors
        idx = np.argsort(D, axis=1)[:, :k]
        dists = np.take_along_axis(D, idx, axis=1)

        # Smooth kNN for new points
        rho, sigma = self._smooth_knn_dist(dists)
        numer = np.exp(-(dists - rho[:, None]) / (sigma[:, None] + 1e-12))
        numer[dists < rho[:, None]] = 1.0
        w = numer / (np.sum(numer, axis=1, keepdims=True) + 1e-12)

        # Weighted average of neighbor embeddings
        Y_new = np.einsum("ij,ijk->ik", w, Y_train[idx])
        return Y_new

    def __str__(self) -> str:
        return "UMAP (Uniform Manifold Approximation and Projection)"
