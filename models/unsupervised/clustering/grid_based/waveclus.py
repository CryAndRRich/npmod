import numpy as np
import pywt
from scipy.ndimage import label
from sklearn.decomposition import PCA
from typing import Optional

class WaveCluster:
    def __init__(self,
                 num_cells: int = 10,
                 wavelet: str = 'haar',
                 wavelet_levels: int = 1,
                 pca_components: Optional[int] = None) -> None:
        """
        WaveCluster algorithm: grid-based clustering using wavelet transforms
        with multi-resolution clustering support and optional PCA for high-dimensional data.

        Parameters:
            num_cells: Number of grid cells per dimension
            wavelet: Type of wavelet to use
            wavelet_levels: Number of wavelet transform levels
            pca_components: Number of PCA components to reduce to (if None, no PCA applied)
        """
        self.num_cells = num_cells
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.pca_components = pca_components
        self.grid = None
        self.bin_edges = []
        self.unit_labels = []  # labels at each resolution
        self.labels_ = None  # final point labels
        self.pca_model = None

    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """
        Apply PCA if pca_components is set and data is high-dimensional
        """
        _, n_features = features.shape
        if self.pca_components is not None and n_features > self.pca_components:
            self.pca_model = PCA(n_components=self.pca_components)
            return self.pca_model.fit_transform(features)
        return features

    def _init_grid(self, features: np.ndarray) -> None:
        """
        Initialize the grid and assign points to grid cells
        """
        _, n_features = features.shape
        self.bin_edges = []
        grid_shape = []

        for d in range(n_features):
            edges = np.linspace(features[:, d].min(), features[:, d].max(), self.num_cells + 1)
            self.bin_edges.append(edges)
            grid_shape.append(self.num_cells)

        self.grid = np.zeros(grid_shape, dtype=int)

        # Assign points to grid
        indices = []
        for d in range(n_features):
            idx = np.searchsorted(self.bin_edges[d], features[:, d], side='right') - 1
            idx[idx == self.num_cells] = self.num_cells - 1
            indices.append(idx)
        indices = np.array(indices).T

        for idx in indices:
            self.grid[tuple(idx)] += 1

    def _wavelet_transform(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply wavelet transform to the grid

        Returns:
            Approximation coefficients after multiple levels
        """
        coeffs = grid
        for _ in range(self.wavelet_levels):
            coeffs = pywt.dwtn(coeffs, self.wavelet)['aa']  # approximation subband
        return coeffs

    def _find_clusters(self, transformed_grid: np.ndarray) -> np.ndarray:
        """
        Find connected components in the transformed grid
        using a threshold based on mean amplitude
        """
        threshold = np.mean(transformed_grid)
        binary_grid = transformed_grid > threshold
        labeled_units, _ = label(binary_grid)
        return labeled_units

    def _assign_labels(self, features: np.ndarray, labeled_units: np.ndarray) -> np.ndarray:
        """
        Map points to clusters based on grid cell assignments
        """
        n_samples, n_features = features.shape
        unit_indices = []

        for d in range(n_features):
            idx = np.searchsorted(self.bin_edges[d], features[:, d], side='right') - 1
            idx[idx == self.num_cells] = self.num_cells - 1
            # Map index to transformed grid size
            scale = labeled_units.shape[d] / self.num_cells
            idx = np.floor(idx * scale).astype(int)
            idx[idx >= labeled_units.shape[d]] = labeled_units.shape[d] - 1
            unit_indices.append(idx)

        unit_indices = np.array(unit_indices).T
        labels = np.zeros(n_samples, dtype=int)

        for i, idx in enumerate(unit_indices):
            labels[i] = labeled_units[tuple(idx)]
        return labels

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit WaveCluster to the input data and perform multi-resolution clustering
        with optional PCA for high-dimensional data

        Parameters:
            features: Data matrix

        Returns:
            labels_: Cluster assignments for each data point at the finest resolution
        """
        features = self._apply_pca(features)
        self._init_grid(features)
        current_grid = self.grid.copy()
        self.unit_labels = []

        for _ in range(self.wavelet_levels):
            transformed_grid = self._wavelet_transform(current_grid)
            labeled_units = self._find_clusters(transformed_grid)
            self.unit_labels.append(labeled_units)
            current_grid = transformed_grid  # for next coarser level

        # Assign final labels at the finest resolution
        self.labels_ = self._assign_labels(features, self.unit_labels[0])
        return self.labels_

    def get_labels_at_level(self, 
                            level: int, 
                            features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get cluster labels at a specific resolution level

        Parameters:
            level: Level index (0=fine, wavelet_levels-1=coarse)
            features: Original data matrix, optional if labels already mapped

        Returns:
            Cluster labels for all points
        """
        if level < 0 or level >= len(self.unit_labels):
            raise ValueError("Level out of range")

        if features is None:
            if self.labels_ is None:
                raise ValueError("No features provided and model not fitted")
            # If PCA was applied, transform original data
            features = self.labels_ if self.pca_model is None else self.pca_model.transform(self.labels_)
        else:
            features = self._apply_pca(features)
        return self._assign_labels(features, self.unit_labels[level])

    def __str__(self) -> str:
        return "WaveCluster (Wavelets in Grid-Based Clustering)"
