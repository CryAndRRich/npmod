from typing import List
import numpy as np
from ..xgboost.xgb_tree import XGTreeRegressor, TreeNode

class CatTreeRegressor(XGTreeRegressor):
    """
    Regression tree for CatBoost (oblivious-style) with native categorical splits.
    Inherits gradient/hessian infrastructure but builds TreeNode structure
    """
    def __init__(self,
                 cat_features: List[int] = None,
                 n_feats: int = None,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 reg_lambda: float = 1.0,
                 gamma: float = 0.0) -> None:
        """
        Initialize CatTreeRegressor

        Parameters:
            cat_features: Indices of categorical features
            n_feats: Number of features to consider per split
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            reg_lambda: L2 regularization on leaf weights
            gamma: Minimum gain required for a split
        """
        super().__init__(n_feats=n_feats,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         reg_lambda=reg_lambda,
                         gamma=gamma)
        self.cat_features = set(cat_features or [])

    def fit(self,
            features: np.ndarray,
            grad: np.ndarray,
            hess: np.ndarray) -> None:
        """
        Fit the CatBoost tree to gradients and hessians

        Parameters:
            features: Training feature matrix 
            grad: First-order gradients g_i for each sample
            hess: Second-order hessians h_i for each sample
        """
        self._grad, self._hess = grad, hess
        n_features = features.shape[1]
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)
        self.root = self._build_tree(features, grad, hess, depth=0)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Traverse the tree to predict leaf weights for each sample

        Parameters:
            test_features: Feature matrix 

        Returns:
            np.ndarray: Array of leaf weight predictions for each sample
        """
        def _traverse(x, node: TreeNode) -> float:
            if node.value is not None:
                return node.value
            val = x[node.feature]
            if getattr(node, "cat_order", None) is not None:
                order = node.cat_order
                comp = order.index(val) if val in order else len(order) 
            else:
                comp = val
            if comp <= node.threshold:
                return _traverse(x, node.left)
            else:
                return _traverse(x, node.right)
        return np.array([_traverse(x, self.root) for x in test_features])
    
    def _build_tree(self,
                    X: np.ndarray,
                    g: np.ndarray,
                    h: np.ndarray,
                    depth: int) -> TreeNode:
        """
        Recursively build the CatBoost regression tree

        Parameters:
            X: Feature matrix at current node
            g: Gradients for current node samples
            h: Hessians for current node samples
            depth: Current depth in the tree

        Returns:
            TreeNode: Root of the constructed subtree
        """
        n_samples, n_features = X.shape
        G_total, H_total = g.sum(), h.sum()

        # stopping
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=self._leaf_weight(G_total, H_total))

        best_gain, best_feat, best_thr, best_order = 0.0, None, None, None
        is_cat_split = False

        for feat in range(n_features):
            if feat >= self.n_feats:
                break
            vals = X[:, feat]

            if feat in self.cat_features:
                cats = np.unique(vals)
                if len(cats) <= 1:
                    continue
                # order categories theo mean gradient/hessian
                ratios = {c: g[vals == c].sum() / (h[vals == c].sum() + 1e-12) for c in cats}
                order = sorted(cats, key=lambda c: ratios[c])
                feat_vals = np.array([order.index(v) for v in vals])

                for thr in range(len(order)-1):
                    left_mask = feat_vals <= thr
                    right_mask = ~left_mask
                    if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                        continue
                    G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                    G_R, H_R = g[right_mask].sum(), h[right_mask].sum()
                    gain = self._gain(G_L, H_L, G_R, H_R, G_total, H_total)
                    if gain > best_gain:
                        best_gain, best_feat, best_thr = gain, feat, thr
                        best_order, is_cat_split = order, True
            else:
                # numerical
                order = None
                feat_vals = vals.astype(float)
                thresholds = np.unique(feat_vals)
                for thr in thresholds:
                    left_mask = feat_vals <= thr
                    right_mask = ~left_mask
                    if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                        continue
                    G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                    G_R, H_R = g[right_mask].sum(), h[right_mask].sum()
                    gain = self._gain(G_L, H_L, G_R, H_R, G_total, H_total)
                    if gain > best_gain:
                        best_gain, best_feat, best_thr = gain, feat, thr
                        best_order, is_cat_split = None, False

        if best_feat is None or best_gain <= 0:
            return TreeNode(value=self._leaf_weight(G_total, H_total))

        vals = X[:, best_feat]
        if is_cat_split:
            indices = np.array([best_order.index(v) if v in best_order else len(best_order) for v in vals])
        else:
            indices = vals.astype(float)

        left_mask = indices <= best_thr
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth+1)
        right = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth+1)

        return TreeNode(feature=best_feat,
                        threshold=float(best_thr),
                        left=left,
                        right=right,
                        cat_order=best_order if is_cat_split else None)