import heapq
import numpy as np

class ISOMAP():
    def __init__(self, 
                 n_neighbors: int = 5, 
                 n_components: int = 2) -> None:
        """
        Initialize ISOMAP model with hyperparameters

        Parameters:
            n_neighbors: Number of nearest neighbors for graph construction
            n_components: Target dimension for embedding
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.embedding_ = None

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance matrix

        Parameters:
            X: Data matrix 

        Returns:
            dist: Distance matrix 
        """
        sum_X = np.sum(X ** 2, axis=1)
        dist2 = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]
        return np.sqrt(np.maximum(dist2, 0))

    def _dijkstra(self, 
                  graph: np.ndarray, 
                  source: int) -> np.ndarray:
        """Compute shortest path from a single source using Dijkstra"""
        n = graph.shape[0]
        dist = np.full(n, np.inf)
        dist[source] = 0
        visited = np.zeros(n, dtype=bool)
        heap = [(0, source)]

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v, w in enumerate(graph[u]):
                if w < np.inf and not visited[v]:
                    new_dist = d + w
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        heapq.heappush(heap, (new_dist, v))
        return dist

    def _largest_connected_component(self, graph: np.ndarray) -> np.ndarray:
        """Find mask for largest connected component using BFS"""
        n = graph.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = []
        
        for start in range(n):
            if not visited[start]:
                queue = [start]
                comp = []
                visited[start] = True
                while queue:
                    u = queue.pop()
                    comp.append(u)
                    neighbors = np.where(graph[u] < np.inf)[0]
                    for v in neighbors:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                components.append(comp)

        largest = max(components, key=len)
        mask = np.zeros(n, dtype=bool)
        mask[largest] = True
        return mask

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the ISOMAP model and return the low-dimensional embedding

        Parameters:
            features: Input data matrix 

        Returns:
            embedding: Low-dimensional representation
        """
        X = features.astype(np.float64)
        n_samples = X.shape[0]

        # Euclidean distance
        dist = self._pairwise_distances(X)

        # Build symmetric kNN graph
        graph = np.full((n_samples, n_samples), np.inf)
        for i in range(n_samples):
            neighbors = np.argsort(dist[i])[1:self.n_neighbors+1]
            for j in neighbors:
                graph[i, j] = dist[i, j]
                graph[j, i] = dist[i, j]
        np.fill_diagonal(graph, 0.0)

        # Dijkstra all-pairs shortest paths
        D = np.array([self._dijkstra(graph, i) for i in range(n_samples)])

        # Replace infinities with large number
        max_dist = np.nanmax(D[D < np.inf])
        D[D == np.inf] = max_dist * 10  

        # Classical MDS
        J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        D2 = D**2
        B = -0.5 * J.dot(D2).dot(J)

        # Eigen-decomposition
        eigen_vals, eigen_vectors = np.linalg.eigh(B)
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vectors = eigen_vectors[:, idx]

        pos_idx = eigen_vals > 1e-12
        eigen_vals = eigen_vals[pos_idx][:self.n_components]
        eigen_vectors = eigen_vectors[:, pos_idx][:, :self.n_components]

        self.embedding_ = eigen_vectors * np.sqrt(eigen_vals)[None, :]
        return self.embedding_

    def __str__(self) -> str:
        return "ISOMAP (Isometric Mapping)"
