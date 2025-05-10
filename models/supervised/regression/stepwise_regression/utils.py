import numpy as np

def compute_mse(preds: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Mean Squared Error between predictions and true values
    """
    return np.mean((preds - y) ** 2)


def compute_r2(preds: np.ndarray, y: np.ndarray) -> float:
    """
    Compute R-squared coefficient of determination
    """
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def compute_aic(n: int, mse: float, k: int) -> float:
    """
    Compute Akaike Information Criterion for a linear model

    AIC = n * ln(RSS/n) + 2k, where RSS = n * mse, k = number of parameters
    """
    rss = mse * n
    return n * np.log(rss / n) + 2 * k