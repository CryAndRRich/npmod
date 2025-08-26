# This script tests various dimension reduction algorithms, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_iris
from sklearn.manifold import trustworthiness

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Importing built-in algorithms from sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as SE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Importing the custom algorithms
from models.unsupervised.dimensionality_reduction.linear.pca import PCA as CustomPCA
from models.unsupervised.dimensionality_reduction.linear.lda import LDA as CustomLDA
from models.unsupervised.dimensionality_reduction.linear.nmf import NMF as CustomNMF

from models.unsupervised.dimensionality_reduction.nonlinear.t_sne import tSNE as CustomTSNE
from models.unsupervised.dimensionality_reduction.nonlinear.lle import LLE as CustomLLE
from models.unsupervised.dimensionality_reduction.nonlinear.mds import MDS as CustomMDS
from models.unsupervised.dimensionality_reduction.nonlinear.isomap import ISOMAP as CustomIsomap
from models.unsupervised.dimensionality_reduction.nonlinear.se import SpectralEmbedding as CustomSE


# === Model information ===
# 1-2: PCA
# 3-4: LDA
# 5-6: NMF
# 7-8: t-SNE
# 9-10: LLE
# 11-12: MDS
# 13-14: Isomap
# 15-16: Spectral Embedding
# ====================


if __name__ == "__main__":
    # === Load all datasets ===
    # Load Iris dataset 
    X_iris, y_iris = load_iris(return_X_y=True)
    X_iris = scaler.fit_transform(X_iris)
    # ====================


    # === PCA using built-in and custom implementations ===
    model1 = PCA(n_components=2)
    model1.fit(X_iris)

    model2 = CustomPCA(n_components=2)
    model2.fit(X_iris)

    print("==============================================================")
    print("PCA Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, model1.transform(X_iris), n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, model2.transform(X_iris), n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9742800495662949
    Trustworthiness (custom): 0.9742800495662949  
    """
    # ====================


    # === LDA using built-in and custom implementations ===
    model3 = LDA(n_components=2)
    X_lda3 = model3.fit_transform(X_iris, y_iris)

    model4 = CustomLDA(n_components=2)
    model4.fit(X_iris, y_iris)
    X_lda4 = model4.transform(X_iris)

    print("==============================================================")
    print("LDA Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_lda3, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_lda4, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9253035935563817
    Trustworthiness (custom): 0.9272366790582404
    """
    # ====================


    # === NMF using built-in and custom implementations ===
    model5 = NMF(n_components=2, solver="mu", init="nndsvda")
    model5.fit(X_iris)

    model6 = CustomNMF(n_components=2, init="nndsvda")
    model6.fit(X_iris)

    print("==============================================================")
    print("NMF Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, model5.transform(X_iris), n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, model6.transform(X_iris), n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9682428748451053
    Trustworthiness (custom): 0.9183048327137546
    """
    # ====================


    # === tSNE using built-in and custom implementations ===
    model7 = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42, init="random")
    X_tsne7 = model7.fit_transform(X_iris)

    model8 = CustomTSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    X_tsne8 = model8.fit_transform(X_iris, CustomPCA(n_components=2))

    print("==============================================================")
    print("tSNE Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_tsne7, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_tsne8, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9872317224287485
    Trustworthiness (custom): 0.980728624535316
    """
    # ====================


    # === LLE using built-in and custom implementations ===
    model9 = LLE(n_neighbors=10, n_components=2, method="standard", random_state=42)
    X_lle9 = model9.fit_transform(X_iris)

    model10 = CustomLLE(n_neighbors=10, n_components=2)
    X_lle10 = model10.fit_transform(X_iris)

    print("==============================================================")
    print("LLE Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_lle9, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_lle10, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.7869838909541511
    Trustworthiness (custom): 0.8126493184634449
    """
    # ====================


    # === MDS using built-in and custom implementations ===
    model11 = MDS(n_components=2, max_iter=300, n_init=4, random_state=42, dissimilarity="euclidean")
    X_mds11 = model11.fit_transform(X_iris)

    model12 = CustomMDS(n_components=2, max_iter=300, n_init=4, random_state=42)
    X_mds12 = model12.fit_transform(X_iris)

    print("==============================================================")
    print("MDS Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_mds11, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_mds12, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9787757125154894
    Trustworthiness (custom): 0.9819677819083024
    """
    # ====================


    # === Isomap using built-in and custom implementations ===
    model13 = Isomap(n_neighbors=20, n_components=2)
    X_iso13 = model13.fit_transform(X_iris)

    model14 = CustomIsomap(n_neighbors=20, n_components=2)
    X_iso14 = model14.fit_transform(X_iris)

    print("==============================================================")
    print("Isomap Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_iso13, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_iso14, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9699727385377943
    Trustworthiness (custom): 0.9700173482032218
    """
    # ====================


    # === Spectral Embedding using built-in and custom implementations ===
    model15 = SE(n_components=2, n_neighbors=30, affinity="nearest_neighbors", random_state=0)
    X_se15 = model15.fit_transform(X_iris)

    model16 = CustomSE(n_components=2, n_neighbors=30, affinity="nearest_neighbors")
    X_se16 = model16.fit_transform(X_iris)

    print("==============================================================")
    print("Spectral Embedding Results")
    print("==============================================================")
    print("Trustworthiness (built-in):", trustworthiness(X_iris, X_se15, n_neighbors=10))
    print("Trustworthiness (custom):", trustworthiness(X_iris, X_se16, n_neighbors=10))

    """
    Trustworthiness (built-in): 0.9378785625774473
    Trustworthiness (custom): 0.9261016109045849
    """
    # ====================
