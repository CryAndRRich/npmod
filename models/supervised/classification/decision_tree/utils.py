from typing import Tuple
import math
import numpy as np

def split_data(features: np.ndarray, 
               targets: np.ndarray, 
               feature: int, 
               value: float, 
               weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset based on a specified feature and its value

    Parameters:
        features: Feature matrix of the data
        targets: Array of targets corresponding to the features
        feature: Index of the feature used for splitting
        value: Threshold value for the feature to split on
        weights: Array of weights corresponding to each sample

    Returns:
        true_features, true_targets, false_features, false_targets: 
            Subsets of features and targets for the two branches
        If weights are provided, returns true_weights and false_weights as well
    """
    if isinstance(value, (int, float)):
        true_indices = np.where(features[:, feature] <= value)[0]
        false_indices = np.where(features[:, feature] > value)[0]
    else:  
        true_indices = np.where(features[:, feature] == value)[0]
        false_indices = np.where(features[:, feature] != value)[0]

    true_features, true_targets = features[true_indices], targets[true_indices]
    false_features, false_targets = features[false_indices], targets[false_indices]

    if weights is None:
        return true_features, true_targets, false_features, false_targets
    
    true_weights, false_weights = weights[true_indices], weights[false_indices]
    return true_features, true_targets, true_weights, false_features, false_targets, false_weights

def entropy(targets: np.ndarray, 
            weights: np.ndarray = None) -> float:
    """
    Computes the entropy of a target distribution

    Parameters:
        targets: Array of targets
        weights: Array of weights corresponding to each target

    Returns:
        float: Entropy of the target distribution
    """
    if weights is None:
        weights = np.ones(len(targets))
    weighted_counts = np.bincount(targets, weights=weights)
    total_weight = np.sum(weights)
    probs = weighted_counts / total_weight
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(true_targets: np.ndarray, 
                     false_targets: np.ndarray, 
                     current_entropy: float, 
                     true_weights: np.ndarray = None, 
                     false_weights: np.ndarray = None, 
                     get_ratio: bool = False) -> float:
    """
    Calculates the information gain from a split

    Parameters:
        true_targets: targets of the left branch
        false_targets: targets of the right branch
        current_entropy: Entropy before the split
        true_weights: Weights of samples in the left branch
        false_weights: Weights of samples in the right branch
        get_ratio: Whether to return the gain ratio

    Returns:
        float: Information gain (or gain ratio if `get_ratio=True`)
    """
    true = len(true_targets) if true_weights is None else np.sum(true_weights)
    false = len(false_targets) if false_weights is None else np.sum(false_weights)

    total_size = true + false
    true_ratio = true / total_size
    false_ratio = false / total_size

    true_entropy = entropy(true_targets, true_weights)
    false_entropy = entropy(false_targets, false_weights)

    information_gain = current_entropy - (true_ratio * true_entropy + false_ratio * false_entropy)
    if not get_ratio:
        return information_gain

    split_info = -(true_ratio * np.log2(true_ratio) + false_ratio * np.log2(false_ratio)) if true_ratio > 0 and false_ratio > 0 else 0
    return information_gain / split_info if split_info != 0 else 0

def gini_impurity(targets: np.ndarray, 
                  weights: np.ndarray = None) -> float:
    """
    Computes the Gini impurity of a target distribution

    Parameters:
        targets: Array of targets
        weights: Array of weights corresponding to each target

    Returns:
        float: Gini impurity of the target distribution
    """
    if weights is None:
        weights = np.ones(len(targets))
    weighted_counts = np.bincount(targets, weights=weights)
    total_weight = np.sum(weights)
    probs = weighted_counts / total_weight
    return 1 - np.sum([p ** 2 for p in probs if p > 0])

def gini_index(true_targets: np.ndarray, 
               false_targets: np.ndarray, 
               current_uncertainty: float, 
               true_weights: np.ndarray = None, 
               false_weights: np.ndarray = None) -> float:
    """
    Calculates the Gini index for a split

    Parameters:
        true_targets: targets of the left branch
        false_targets: targets of the right branch
        current_uncertainty:  Gini impurity before the split
        true_weights: Weights of samples in the left branch
        false_weights: Weights of samples in the right branch

    Returns:
        float: Reduction in Gini impurity from the split
    """
    true = len(true_targets) if true_weights is None else np.sum(true_weights)
    false = len(false_targets) if false_weights is None else np.sum(false_weights)

    total_size = true + false
    true_ratio = true / total_size
    false_ratio = false / total_size

    true_entropy = gini_impurity(true_targets, true_weights)
    false_entropy = gini_impurity(false_targets, false_weights)

    return current_uncertainty - (true_ratio * true_entropy + false_ratio * false_entropy)

def chi_square(true_targets: np.ndarray, 
               false_targets: np.ndarray, 
               total_targets: np.ndarray, 
               true_weights: np.ndarray = None, 
               false_weights: np.ndarray = None) -> float:
    """
    Computes the chi-square statistic for a split

    Parameters:
        true_targets: targets of the left branch
        false_targets: targets of the right branch
        total_targets: Total targets before the split
        true_weights: Weights of samples in the left branch
        false_weights: Weights of samples in the right branch

    Returns:
        chi_square_stat: Chi-square statistic
    """
    if true_weights is None:
        true_weights = np.ones(len(true_targets))
    if false_weights is None:
        false_weights = np.ones(len(false_targets))

    num_classes = len(np.bincount(total_targets))

    true_counts = np.bincount(true_targets, weights=true_weights, minlength=num_classes)
    false_counts = np.bincount(false_targets, weights=false_weights, minlength=num_classes)
    total_counts = np.bincount(total_targets, minlength=num_classes)

    total_true = np.sum(true_counts)
    total_false = np.sum(false_counts)
    total_all = total_true + total_false

    chi_square_stat = 0.0
    for observed_true, observed_false, total_count in zip(true_counts, false_counts, total_counts):
        expected_true = total_count * (total_true / total_all) if total_all > 0 else 0
        expected_false = total_count * (total_false / total_all) if total_all > 0 else 0

        if expected_true > 0:
            chi_square_stat += ((observed_true - expected_true) ** 2) / expected_true
        if expected_false > 0:
            chi_square_stat += ((observed_false - expected_false) ** 2) / expected_false

    return chi_square_stat

def chi_square_p_value(chi_square_stat: float, 
                       df: int) -> float:
    """
    Calculates the p-value for a given chi-square statistic and degrees of freedom

    Parameters:
        chi_square_stat: Chi-square statistic
        df: Degrees of freedom

    Returns:
        float: p-value for the chi-square statistic
    """
    
    def lower_incomplete_gamma(s, x):
        """Computes the lower incomplete gamma function P(s, x)"""
        result = 0
        eps = 1e-9
        term = 1 / (s + eps) # First term of the series expansion
        for k in range(1, 100):  # Iterate for convergence (adjust max iterations if needed)
            term *= x / (s + k)
            result += term
            if term < 1e-10:  # Convergence threshold
                break
        return (math.exp(-x) * (x ** s) / math.gamma(s + eps)) * (1 + result)

    # Compute the upper tail probability (survival function)
    p_value = 1 - lower_incomplete_gamma(df / 2, chi_square_stat / 2)
    
    return max(min(p_value, 1.0), 0.0)  # Ensure p-value is in [0, 1]
