# This script tests various custom clustering models
# The results can be seen in data/img/result_cluster.png in case you don't want to run the code

from itertools import cycle, islice
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Importing the custom model
from partition_based import KMeans
from tree_based import AgglomerativeClustering, DivisiveClustering, CURE, CHAMELEON
from graph_based import SpectralClustering
from grid_based import GRIDCLUS, STING, WaveCluster
from density_based import DBSCAN, OPTICS
from model_based import GaussianMixtureModel, BayesianGaussianMixtureModel


def make_three_circles(n_samples=1500, noise=0.05, width=0.2):
    np.random.seed(0)
    n_samples_out = n_samples // 3
    n_samples_mid = n_samples // 3
    n_samples_in = n_samples - n_samples_out - n_samples_mid

    radii = [1, 2, 3]

    def make_ring(n, r):
        angles = 2 * np.pi * np.random.rand(n)
        rs = r + width * (np.random.rand(n) - 0.5)
        x = rs * np.cos(angles) + noise * np.random.randn(n)
        y = rs * np.sin(angles) + noise * np.random.randn(n)
        return x, y

    outer_x, outer_y = make_ring(n_samples_out, radii[2])
    mid_x, mid_y = make_ring(n_samples_mid, radii[1])
    inner_x, inner_y = make_ring(n_samples_in, radii[0])

    X = np.vstack([
        np.stack([outer_x, outer_y], axis=1),
        np.stack([mid_x, mid_y], axis=1),
        np.stack([inner_x, inner_y], axis=1)
    ])
    y = np.array([0] * n_samples_out + [1] * n_samples_mid + [2] * n_samples_in)

    return X, y


def make_spiral(n_points=1500, noise=0.03, n_classes=2):
    np.random.seed(0)
    X = []
    y = []
    n = n_points // n_classes
    
    for j in range(n_classes):
        theta = np.linspace(0, 4 * np.pi, n) + (2 * np.pi * j / n_classes)
        r = np.linspace(0.1, 1, n) 
        x = r * np.cos(theta) + noise * np.random.randn(n)
        y_points = r * np.sin(theta) + noise * np.random.randn(n)
        X += list(zip(x, y_points))
        y += [j] * n
    
    return np.array(X), np.array(y)


n_samples = 500
seed = 30
random_state = 170
transformation = [[0.6, -0.6], [-0.4, 0.8]]

circles = make_three_circles(n_samples=2*n_samples, width=0.2)

moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

spiral = make_spiral(n_points=2*n_samples, noise=0.02)

rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

datasets_list = [
    ("Blobs", blobs, {
        "KMeans": {"k": 3}, 
        
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
        "DivisiveClustering": {"n_clusters": 3},
        "CURE": {"n_clusters": 3},
        "CHAMELEON": {"n_clusters": 3},
        
        "DBSCAN": {"eps": 0.2, "min_samples": 5},
        "OPTICS": {"max_eps": 0.2},
        
        "GridClus": {"grid_shape": (19, 19), "min_points": 3},
        "STING": {"max_level": 4, "min_points": 6},
        "WaveCluster": {"num_cells": 10},
        
        "SpectralClustering": {"n_clusters": 3},
        
        "GaussianMixtureModel": {"k": 3},
        "BayesianGaussianMixtureModel": {"k": 3}
        }),

    ("Varied blobs", varied, {
        "KMeans": {"k": 3},
        
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
        "DivisiveClustering": {"n_clusters": 3},
        "CURE": {"n_clusters": 3,},
        "CHAMELEON": {"n_clusters": 3},
        
        "DBSCAN": {"eps": 0.2, "min_samples": 4},
        "OPTICS": {"max_eps": 0.2},
        
        "GridClus": {"grid_shape": (12, 12), "min_points": 4},
        "STING": {"max_level": 4, "min_points": 3},
        "WaveCluster": {"num_cells": 9},
        
        "SpectralClustering": {"n_clusters": 3},
        
        "GaussianMixtureModel": {"k": 3},
        "BayesianGaussianMixtureModel": {"k": 3}
        }),

    ("Circles", circles, {
        "KMeans": {"k": 3},
        
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "single"},
        "DivisiveClustering": {"n_clusters": 3},
        "CURE": {"n_clusters": 3},
        "CHAMELEON": {"n_clusters": 3},
        
        "DBSCAN": {"eps": 0.3, "min_samples": 4},
        "OPTICS": {"max_eps": 0.3},
        
        "GridClus": {"grid_shape": (17, 17), "min_points": 3},
        "STING": {"max_level": 5, "min_points": 1},
        "WaveCluster": {"num_cells": 6},
        
        "SpectralClustering": {"n_clusters": 3},
        
        "GaussianMixtureModel": {"k": 3},
        "BayesianGaussianMixtureModel": {"k": 3}
        }),

    ("Moons", moons, {
        "KMeans": {"k": 2},
        
        "AgglomerativeClustering": {"n_clusters": 2, "linkage": "single"},
        "DivisiveClustering": {"n_clusters": 2},
        "CURE": {"n_clusters": 2},
        "CHAMELEON": {"n_clusters": 2},
        
        "DBSCAN": {"eps": 0.3, "min_samples": 4},
        "OPTICS": {"max_eps": 0.3},
        
        "GridClus": {"grid_shape": (20, 20), "min_points": 3},
        "STING": {"max_level": 4, "min_points": 3},
        "WaveCluster": {"num_cells": 6},
        
        "SpectralClustering": {"n_clusters": 2},
        
        "GaussianMixtureModel": {"k": 2},
        "BayesianGaussianMixtureModel": {"k": 2}
        }),

    ("Spiral", spiral, {
        "KMeans": {"k": 2},
        
        "AgglomerativeClustering": {"n_clusters": 2, "linkage": "single"},
        "DivisiveClustering": {"n_clusters": 2},
        "CURE": {"n_clusters": 2},
        "CHAMELEON": {"n_clusters": 2},
        
        "DBSCAN": {"eps": 0.2, "min_samples": 4},
        "OPTICS": {"max_eps": 0.2},
        
        "GridClus": {"grid_shape": (38, 38), "min_points": 1},
        "STING": {"max_level": 4, "min_points": 6},
        "WaveCluster": {"num_cells": 6},
        
        "SpectralClustering": {"n_clusters": 2},
        
        "GaussianMixtureModel": {"k": 2},
        "BayesianGaussianMixtureModel": {"k": 2}
        }),

    ("Anisotropic", aniso, {
        "KMeans": {"k": 3},
        
        "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
        "DivisiveClustering": {"n_clusters": 3},
        "CURE": {"n_clusters": 3},
        "CHAMELEON": {"n_clusters": 3},
        
        "DBSCAN": {"eps": 0.2, "min_samples": 4},
        "OPTICS": {"max_eps": 0.2},
        
        "GridClus": {"grid_shape": (24, 24), "min_points": 3},
        "STING": {"max_level": 4, "min_points": 6},
        "WaveCluster": {"num_cells": 12},
        
        "SpectralClustering": {"n_clusters": 3},
        
        "GaussianMixtureModel": {"k": 3},
        "BayesianGaussianMixtureModel": {"k": 3}
        }),

    ("No structure", no_structure, {
        "KMeans": {"k": 4},
        
        "AgglomerativeClustering": {"n_clusters": 4, "linkage": "ward"},
        "DivisiveClustering": {"n_clusters": 4},
        "CURE": {"n_clusters": 4},
        "CHAMELEON": {"n_clusters": 4},
        
        "DBSCAN": {"eps": 0.5, "min_samples": 3},
        "OPTICS": {"max_eps": 0.3},
        
        "GridClus": {"grid_shape": (15, 15), "min_points": 3},
        "STING": {"max_level": 3, "min_points": 5},
        "WaveCluster": {"num_cells": 6},
        
        "SpectralClustering": {"n_clusters": 4},
        
        "GaussianMixtureModel": {"k": 4},
        "BayesianGaussianMixtureModel": {"k": 4}
        }),
]

clustering_algorithms = [
    ("KMeans", KMeans()),
    ("AgglomerativeClustering", AgglomerativeClustering()),
    ("DivisiveClustering", DivisiveClustering()),
    ("CURE", CURE()),
    ("CHAMELEON", CHAMELEON()),
    ("DBSCAN", DBSCAN()),
    ("OPTICS", OPTICS()),
    ("GridClus", GRIDCLUS()),
    ("STING", STING()),
    ("WaveCluster", WaveCluster()),
    ("SpectralClustering", SpectralClustering()),
    ("GaussianMixtureModel", GaussianMixtureModel()),
    ("BayesianGaussianMixtureModel", BayesianGaussianMixtureModel()),
]

n_rows = len(clustering_algorithms) + 1  
n_cols = len(datasets_list)

plt.figure(figsize=(3 * n_cols, 3 * n_rows))

for col, (ds_name, dataset, ds_kwargs) in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    plt.subplot(n_rows, n_cols, 1 + col)
    plt.scatter(X[:, 0], X[:, 1], s=10, color="black")
    if col == 0:
        plt.ylabel("Input", size=15)
    plt.title(ds_name, size=14)
    plt.xticks(())
    plt.yticks(())
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    for row, (name, algorithm) in enumerate(clustering_algorithms, start=1):
        for k, v in ds_kwargs[name].items():
            setattr(algorithm, k, v)

        y_pred = algorithm.fit(X)

        plt.subplot(n_rows, n_cols, row*n_cols + col + 1)
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        colors = np.append(colors, ["#000000"])  
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        if col == 0:
            plt.ylabel(name, size=15)
        plt.xticks(())
        plt.yticks(())
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

plt.tight_layout()
plt.show()
