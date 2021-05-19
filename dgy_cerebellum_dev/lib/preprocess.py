import numpy as np
import scipy.stats as stats

#------------------------------------------------
# Finding out data outliers and further operation

def get_iqr(x: np.ndarray, n: float = 3) -> tuple:
    """Calculates bounds of n * IQR of given array.

    Args:
        x (np.ndarray): 1-D array.
        n (float, optional): Defaults to 3.

    Returns:
        tuple: (lower_bound, upper_bound).
    """
    Q1, Q3 = stats.scoreatpercentile(x, 25), stats.scoreatpercentile(x, 75)
    assert not np.isinf(Q1) and not np.isinf(Q3)
    IQR = Q3 - Q1
    l, u = Q1 - n * IQR, Q3 + n * IQR
    return l, u

def niqr_outlier_indices(x: np.ndarray, n: float = 3) -> tuple:
    """For a given `np.ndarray`, finds out indices of outliers beyond n * IQR.

    Args:
        x (np.ndarray): 1-D array.
        n (float, optional): Defaults to 3.

    Returns:
        tuple: Indices.
    """
    l, u = get_iqr(x, n)
    return np.where((x - l) * (x - u) > 0)

def bound_outlier_indices(x: np.ndarray, l: float, u: float) -> tuple:
    """For a 1-D array, finds out indices of outliers beyond given bounds.

    Args:
        x (np.ndarray): 1-D array.
        l (float): Lower bound.
        u (float): Upper bound.

    Returns:
        tuple: Indices.
    """
    return np.where((x - l) * (x - u) > 0)[0]

#---------------------
# Scaling data

def rescale(x: np.ndarray, l: float, u: float) -> np.ndarray:
    """Rescales given data to interval [l, u].

    Args:
        x (np.ndarray): 1-D array.
        l (float): Lower bound.
        u (float): Upper bound.

    Returns:
        np.ndarray: Rescaled array.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler((l, u))
    return scaler.fit_transform(x)