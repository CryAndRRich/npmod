import numpy as np
import heapq
from typing import List, Dict

class CURE:
    def __init__(self,
                 number_of_clusters: int = 2,
                 num_representative_points: int = 10,
                 shrink_factor: float = 0.3) -> None:
        """
        CURE (Clustering Using REpresentatives) algorithm

        Parameters:
            number_of_clusters: The number of clusters to form
            num_representative_points: Number of representative points to use for each cluster
            shrink_factor: Factor (0 < alpha <= 1) to shrink representative points toward cluster centroid
        """
        if not (0 < shrink_factor <= 1):
            raise ValueError("shrink_factor must be in (0, 1].")

        self.n_clusters = number_of_clusters
        self.n_repr = num_representative_points
        self.shrink_factor = shrink_factor

        self.labels_ = None
        self.representatives_ = None
        self.centroids_ = None

    def _pairwise_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two points"""
        return np.linalg.norm(a - b)

    def _cluster_distance(self, reps_a: np.ndarray, reps_b: np.ndarray) -> float:
        """Compute cluster-to-cluster distance based on nearest representative points"""
        return np.min(np.linalg.norm(reps_a[:, None, :] - reps_b[None, :, :], axis=2))

    def _choose_representatives(self, points: np.ndarray) -> np.ndarray:
        """Select representative points spread out within a cluster"""
        if len(points) <= self.n_repr:
            reps = points.copy()
        else:
            reps = []
            idx = np.random.randint(len(points))
            reps.append(points[idx])

            for _ in range(1, self.n_repr):
                dists = np.min(
                    np.linalg.norm(points[:, None, :] - np.array(reps)[None, :, :], axis=2),
                    axis=1
                )
                idx = np.argmax(dists)
                reps.append(points[idx])
            reps = np.array(reps)

        centroid = np.mean(points, axis=0)
        reps = centroid + self.shrink_factor * (reps - centroid)
        return reps

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Perform CURE clustering on the data

        Parameters:
            features: Feature matrix of the training data

        Returns:
            labels: Cluster labels for each sample (0 to k-1)
        """
        n = features.shape[0]
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters must be in [1, n_samples]")

        clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
        representatives: Dict[int, np.ndarray] = {
            i: self._choose_representatives(features[[i]])
            for i in range(n)
        }

        # build initial heap
        heap = []
        for i in range(n):
            for j in range(i + 1, n):
                d = self._cluster_distance(representatives[i], representatives[j])
                heapq.heappush(heap, (d, i, j))

        active = set(range(n))
        next_id = n

        # iterative merging
        while len(active) > self.n_clusters:
            while True:
                if not heap:
                    raise RuntimeError("Heap ran empty before reaching target clusters.")
                d, a, b = heapq.heappop(heap)
                if (a in active) and (b in active):
                    break

            # merge clusters
            merged_indices = clusters[a] + clusters[b]
            merged_points = features[merged_indices]
            new_reps = self._choose_representatives(merged_points)

            clusters[next_id] = merged_indices
            representatives[next_id] = new_reps

            active.remove(a)
            active.remove(b)
            active.add(next_id)

            # update distances
            for w in list(active):
                if w == next_id:
                    continue
                dist = self._cluster_distance(new_reps, representatives[w])
                heapq.heappush(heap, (dist, next_id, w))

            next_id += 1

        # assign labels
        labels = np.empty(n, dtype=int)
        for label, cid in enumerate(active):
            labels[clusters[cid]] = label

        # save attributes
        self.labels_ = labels
        self.representatives_ = [representatives[cid] for cid in active]
        self.centroids_ = np.array([np.mean(features[clusters[cid]], axis=0) for cid in active])

        return labels

    def __str__(self) -> str:
        return "CURE (Clustering Using REpresentatives)"
