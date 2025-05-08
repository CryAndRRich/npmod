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
                 gamma: float = 0.0) -> None:
        """
        Initialize the LightGBM tree regressor

        Parameters:
            n_feats: Maximum number of features to consider when looking for splits
            max_leaves: Maximum number of leaves to grow (tree complexity control)
            min_samples_split: Minimum number of samples required in a leaf to consider splitting
            reg_lambda: L2 regularization term for leaf weight (λ)
            gamma: Minimum loss reduction (gain) required to make a split (γ)
        """
        super().__init__(n_feats=n_feats,
                         max_depth=None,          
                         min_samples_split=min_samples_split,
                         reg_lambda=reg_lambda,
                         gamma=gamma)
        self.max_leaves = max_leaves

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
            features: Training feature matrix of shape (n_samples, n_features)
            grad: First-order gradients (g_i) for each sample (shape n_samples,)
            hess: Second-order hessians (h_i) for each sample (shape n_samples,)
        """
        
        _, n_features = features.shape
        self.n_feats = n_features if self.n_feats is None else min(self.n_feats, n_features)

        # Initial root
        G_root, H_root = grad.sum(), hess.sum()
        root_weight = self._leaf_weight(G_root, H_root)
        root = TreeNode(value=root_weight)
        # Heap of (-gain, node, indices, G, H)
        heap = []
        # Compute root's best split
        self._try_push_split(root, features, grad, hess, heap)

        leaves = 1
        while leaves < self.max_leaves and heap:
            # Pick the leaf with highest gain
            neg_gain, node, idxs, _, _, feat, thr = heapq.heappop(heap)
            gain = -neg_gain
            if gain <= 0:
                break

            # Actually split this node
            node.feature = feat
            node.threshold = thr
            node.value = None

            # Child data subsets
            left_idx = idxs[features[idxs, feat] <= thr]
            right_idx = idxs[features[idxs, feat] >  thr]

            # Build children
            G_L, H_L = grad[left_idx].sum(), hess[left_idx].sum()
            G_R, H_R = grad[right_idx].sum(), hess[right_idx].sum()
            left_node  = TreeNode(value=self._leaf_weight(G_L, H_L))
            right_node = TreeNode(value=self._leaf_weight(G_R, H_R))
            node.left, node.right = left_node, right_node

            # Try to split each new leaf
            self._try_push_split(left_node, features[left_idx], grad[left_idx], hess[left_idx],
                                 heap, global_idxs=left_idx)
            self._try_push_split(right_node, features[right_idx], grad[right_idx], hess[right_idx],
                                 heap, global_idxs=right_idx)

            leaves += 1

        self.root = root

    def _try_push_split(self,
                        node: TreeNode,
                        feats: np.ndarray,
                        grad: np.ndarray,
                        hess: np.ndarray,
                        heap: list,
                        global_idxs: np.ndarray = None) -> None:
        """
        Evaluate all splits for a given leaf node and push the best one into the heap

        Parameters:
            node: The leaf TreeNode to consider splitting
            feats: Feature sub-matrix for samples in this leaf (n_leaf, n_features)
            grad: Gradient vector for these samples
            hess: Hessian vector for these samples
            heap: The global heap of candidate splits (modified in place)
            global_idxs: Original dataset row indices corresponding to this leaf
        """
        n_samples, _ = feats.shape
        if n_samples < self.min_samples_split:
            return

        G_tot, H_tot = grad.sum(), hess.sum()
        best_gain, best_feat, best_thr = 0.0, None, None

        for feat in range(feats.shape[1]):
            if feat >= self.n_feats:
                break
            col = feats[:, feat]
            for thr in np.unique(col):
                left = col <= thr
                right = ~left
                if left.sum() < self.min_samples_split or right.sum() < self.min_samples_split:
                    continue

                G_L, H_L = grad[left].sum(), hess[left].sum()
                G_R, H_R = grad[right].sum(), hess[right].sum()
                gain = self._gain(G_L, H_L, G_R, H_R, G_tot, H_tot)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr

        if best_feat is not None and best_gain > 0:
            # Store negative gain so heapq gives us max gain first
            idxs = global_idxs if global_idxs is not None else np.arange(n_samples)
            heapq.heappush(heap, (-best_gain, node, idxs, G_tot, H_tot, best_feat, best_thr))