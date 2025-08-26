import heapq
import numpy as np
from ..xgboost.xgb_tree import XGTreeRegressor, TreeNode

class LightGBMTreeRegressor(XGTreeRegressor):
    """
    Leaf-wise regression tree for LightGBM: always split the leaf
    with the highest gain until max_leaves is reached
    """
    def __init__(self,
                 n_feats: int = None,
                 max_leaves: int = 31,
                 min_samples_split: int = 20,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0,
                 feature_fraction: float = 1.0,
                 random_state: int = 42) -> None:
        """
        Initialize the LightGBM tree regressor

        Parameters:
            n_feats: Maximum number of features to consider when looking for splits
            max_leaves: Maximum number of leaves to grow (tree complexity control)
            min_samples_split: Minimum number of samples required in a leaf to consider splitting
            reg_lambda: L2 regularization term for leaf weight
            gamma: Minimum loss reduction (gain) required to make a split
            feature_fraction: Fraction of features to consider at each split
            random_state: Random seed for reproducibility
        """
        super().__init__(n_feats=n_feats,
                         max_depth=None,          
                         min_samples_split=min_samples_split,
                         reg_lambda=reg_lambda,
                         gamma=gamma)
        self.max_leaves = max_leaves
        self.feature_fraction = feature_fraction
        self.rng = np.random.RandomState(random_state)

    def fit(self,
            features: np.ndarray,
            grad: np.ndarray,
            hess: np.ndarray) -> None:
        """
        Build the LightGBM leaf-wise tree using gradients and hessians, 
        maintaining a max-heap of candidate splits, always choosing the split 
        with the highest gain to expand until the "max_leaves" limit is reached 
        or no positive-gain split remains

        Parameters:
            features: Training feature matrix 
            grad: First-order gradients (g_i) for each sample
            hess: Second-order hessians (h_i) for each sample
        """
        
        _, n_features = features.shape
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)

        # Root node
        G_root, H_root = grad.sum(), hess.sum()
        root_weight = self._leaf_weight(G_root, H_root)
        root = TreeNode(value=root_weight)

        heap = []
        self._try_push_split(root, features, grad, hess, heap, global_idxs=np.arange(len(grad)))

        leaves = 1
        while leaves < self.max_leaves and heap:
            neg_gain, node, idxs, _, _, feat, thr = heapq.heappop(heap)
            gain = -neg_gain
            if gain <= self.gamma:  # must exceed gamma
                break

            # Split node
            node.feature = feat
            node.threshold = thr
            node.value = None

            left_mask = features[idxs, feat] <= thr
            left_idx, right_idx = idxs[left_mask], idxs[~left_mask]

            G_L, H_L = grad[left_idx].sum(), hess[left_idx].sum()
            G_R, H_R = grad[right_idx].sum(), hess[right_idx].sum()

            left_node = TreeNode(value=self._leaf_weight(G_L, H_L))
            right_node = TreeNode(value=self._leaf_weight(G_R, H_R))
            node.left, node.right = left_node, right_node

            # Try split children
            self._try_push_split(left_node, features, grad, hess, heap, global_idxs=left_idx)
            self._try_push_split(right_node, features, grad, hess, heap, global_idxs=right_idx)

            leaves += 1

        self.root = root

    def _try_push_split(self,
                        node: TreeNode,
                        features: np.ndarray,
                        grad: np.ndarray,
                        hess: np.ndarray,
                        heap: list,
                        global_idxs: np.ndarray = None) -> None:
        """
        Evaluate all splits for a given leaf node and push the best one into the heap

        Parameters:
            node: The leaf TreeNode to consider splitting
            features: Feature sub-matrix for samples in this leaf (n_leaf, n_features)
            grad: Gradient vector for these samples
            hess: Hessian vector for these samples
            heap: The global heap of candidate splits (modified in place)
            global_idxs: Original dataset row indices corresponding to this leaf
        """
        n_samples = len(global_idxs)
        if n_samples < self.min_samples_split:
            return

        G_tot, H_tot = grad[global_idxs].sum(), hess[global_idxs].sum()
        best_gain, best_feat, best_thr = 0.0, None, None

        # Random feature subset (like LightGBM feature_fraction)
        feat_indices = self.rng.choice(features.shape[1], 
                                       size=int(self.n_feats * self.feature_fraction),
                                       replace=False)

        for feat in feat_indices:
            col = features[global_idxs, feat]
            order = np.argsort(col)
            col_sorted = col[order]
            grad_sorted = grad[global_idxs][order]
            hess_sorted = hess[global_idxs][order]

            # prefix sums
            G_prefix = np.cumsum(grad_sorted)
            H_prefix = np.cumsum(hess_sorted)

            for i in range(n_samples - 1):
                if col_sorted[i] == col_sorted[i + 1]:
                    continue  # skip identical values
                G_L, H_L = G_prefix[i], H_prefix[i]
                G_R, H_R = G_tot - G_L, H_tot - H_L
                if H_L < 1e-6 or H_R < 1e-6:  # prevent nan
                    continue
                if i + 1 < self.min_samples_split or n_samples - (i + 1) < self.min_samples_split:
                    continue
                gain = self._gain(G_L, H_L, G_R, H_R, G_tot, H_tot)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = (col_sorted[i] + col_sorted[i + 1]) / 2.0

        if best_feat is not None and best_gain > self.gamma:
            heapq.heappush(heap, (-best_gain, node, global_idxs, G_tot, H_tot, best_feat, best_thr))