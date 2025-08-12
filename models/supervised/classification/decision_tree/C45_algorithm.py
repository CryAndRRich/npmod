import numpy as np
from .tree import *
from .utils import entropy, information_gain, split_data

class C45DecisionTree(Tree):
    def __init__(self, 
                 min_samples_split=2, 
                 max_depth=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, 
                   features: np.ndarray, 
                   targets: np.ndarray, 
                   depth: int = 0) -> TreeNode:
        if len(set(targets)) == 1:  
            return TreeNode(results=targets[0])
        if len(targets) < self.min_samples_split:
            return TreeNode(results=np.argmax(np.bincount(targets)))
        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(results=np.argmax(np.bincount(targets)))

        best_gain_ratio = 0
        best_criteria = None
        best_sets = None
        _, n_features = features.shape
        current_entropy = entropy(targets)

        for feature in range(n_features):
            col_values = features[:, feature]

            non_missing_mask = ~np.isnan(col_values)
            if np.sum(non_missing_mask) == 0:
                continue

            unique_values = np.unique(col_values[non_missing_mask])

            if np.issubdtype(col_values.dtype, np.number) and len(unique_values) > 1:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                thresholds = unique_values

            for value in thresholds:
                mask_true = col_values <= value if np.issubdtype(col_values.dtype, np.number) else col_values == value
                mask_false = ~mask_true

                if np.any(np.isnan(col_values)):
                    missing_mask = np.isnan(col_values)
                    true_ratio = np.sum(mask_true & ~missing_mask) / np.sum(~missing_mask)
                    mask_true |= (missing_mask & (np.random.rand(len(targets)) < true_ratio))
                    mask_false |= (missing_mask & ~mask_true)

                true_features, true_targets = features[mask_true], targets[mask_true]
                false_features, false_targets = features[mask_false], targets[mask_false]

                if len(true_targets) == 0 or len(false_targets) == 0:
                    continue

                gain_ratio_value = information_gain(true_targets, false_targets, current_entropy, get_ratio=True)

                if gain_ratio_value > best_gain_ratio:
                    best_gain_ratio = gain_ratio_value
                    best_criteria = (feature, value)
                    best_sets = (true_features, true_targets, false_features, false_targets)

        if best_gain_ratio > 0 and best_sets is not None:
            true_branch = self.build_tree(best_sets[0], best_sets[1], depth + 1)
            false_branch = self.build_tree(best_sets[2], best_sets[3], depth + 1)
            return TreeNode(feature=best_criteria[0],
                            value=best_criteria[1],
                            true_branch=true_branch,
                            false_branch=false_branch)

        return TreeNode(results=np.argmax(np.bincount(targets)))

    def __str__(self):
        return "C4.5 Algorithm"
