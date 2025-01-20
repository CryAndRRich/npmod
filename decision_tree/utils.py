import math
import numpy as np

def split_data(features, labels, feature, value, weights=None):
    if isinstance(value, (int, float)):
        true_indices = np.where(features[:, feature] <= value)[0]
        false_indices = np.where(features[:, feature] > value)[0]
    else:  
        true_indices = np.where(features[:, feature] == value)[0]
        false_indices = np.where(features[:, feature] != value)[0]

    true_features, true_labels = features[true_indices], labels[true_indices]
    false_features, false_labels = features[false_indices], labels[false_indices]

    if weights is None:
        return true_features, true_labels, false_features, false_labels
    
    true_weights, false_weights = weights[true_indices], weights[false_indices]
    return true_features, true_labels, true_weights, false_features, false_labels, false_weights

def entropy(labels, weights=None):
    if weights is None:
        weights = np.ones(len(labels))
    weighted_counts = np.bincount(labels, weights=weights)
    total_weight = np.sum(weights)
    probs = weighted_counts / total_weight
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
    return entropy

def information_gain(true_labels, false_labels, current_entropy, true_weights=None, false_weights=None, get_ratio=False):
    true = len(true_labels) if true_weights is None else np.sum(true_weights)
    false = len(false_labels) if false_weights is None else np.sum(false_weights)

    total_size = true + false
    true_ratio = true / total_size
    false_ratio = false / total_size
    
    true_entropy = entropy(true_labels, true_weights)
    false_entropy = entropy(false_labels, false_weights)
    
    information_gain = current_entropy - (true_ratio * true_entropy + false_ratio * false_entropy)
    if not get_ratio:
        return information_gain
    
    split_info = - (true_ratio * np.log2(true_ratio) + false_ratio * np.log2(false_ratio)) if true_ratio > 0 and false_ratio > 0 else 0
    
    return information_gain / split_info if split_info != 0 else 0

def gini_impurity(labels, weights=None):
    if weights is None:
        weights = np.ones(len(labels))
    weighted_counts = np.bincount(labels, weights=weights)
    total_weight = np.sum(weights)
    probs = weighted_counts / total_weight
    gini = -np.sum([p ** 2 for p in probs if p > 0])
    return gini

def gini_index(true_labels, false_labels, current_uncertainty, true_weights=None, false_weights=None):
    true = len(true_labels) if true_weights is None else np.sum(true_weights)
    false = len(false_labels) if false_weights is None else np.sum(false_weights)

    total_size = true + false
    true_ratio = true / total_size
    false_ratio = false / total_size
    
    true_entropy = gini_impurity(true_labels, true_weights)
    false_entropy = gini_impurity(false_labels, false_weights)
    
    gini_index = current_uncertainty - (true_ratio * true_entropy + false_ratio * false_entropy)
    return gini_index

def chi_square(true_labels, false_labels, total_labels, true_weights=None, false_weights=None):
    true = np.bincount(true_labels, weights=true_weights)
    false = np.bincount(false_labels, weights=false_weights)
    total = np.bincount(total_labels)

    chi_square_stat = 0
    for observed_true, observed_false, total_count in zip(true, false, total):
        expected_true = total_count * sum(true) / (sum(true) + sum(false))
        expected_false = total_count * sum(false) / (sum(true) + sum(false))
        
        if expected_true > 0:
            chi_square_stat += ((observed_true - expected_true) ** 2) / expected_true
        if expected_false > 0:
            chi_square_stat += ((observed_false - expected_false) ** 2) / expected_false
            
    return chi_square_stat

def chi_square_p_value(chi_square_stat, df):
    def upper_incomplete_gamma(s, x):
        eps = 10e-4
        return np.exp(-x + eps) * np.sum([x ** k / math.factorial(k) for k in range(int(s))])

    def regularized_gamma(s, x):
        complete_gamma = math.gamma(s)
        return upper_incomplete_gamma(s, x) / complete_gamma

    return regularized_gamma(df / 2, chi_square_stat / 2)
