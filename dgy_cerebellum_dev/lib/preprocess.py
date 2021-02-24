import numpy as np
import scipy.stats as stats

#------------------------------------------------
# Finding out data outliers and further operation

def niqr_outlier_indices(x, n=2):
    """For a 1-D array, find out indices of outliers beyond n * IQR.
    """
    Q1, Q3 = stats.scoreatpercentile(x, 25), stats.scoreatpercentile(x, 75)
    IQR = Q3 - Q1
    l, u = Q1 - n * IQR, Q3 + n * IQR
    return np.where((x - l) * (x - u) > 0)[0]

def bound_outlier_indices(x, l, u):
    """For a 1-D array, find out indices of outliers beyond given bounds.
    """
    return np.where((x - l) * (x - u) > 0)[0]

def nan_outlier_indices(x):
    return np.where(np.isnan(x))[0]

def get_norm_indices(array: np.ndarray, _outlier_func=niqr_outlier_indices, **kargs):
    """For a 2-D array, find out outlier indices in each row, and combine them into a list.
    """
    outliers = [_outlier_func(x, **kargs) for x in array]
    combined = set()
    for outlier in outliers:
        combined |= set(outlier)
    
    noramls = list(set(np.arange(array.shape[1])) - combined)
    print(f'Remaining {len(noramls)} / {len(array[0])}')
    return noramls

#--------------------------------------------------------------
# Used in processing myelination maps in old-fashion data structure

def set_outliers_to_nan(maps: dict, _method=niqr_outlier_indices, *args):
    """For a dict containing maps of all subjects of all ages, set outliers to nan within subject data.

    Args:
        maps (dict): Expected to be {age: np.ndarray([map, map, ..., map])}.

    Returns:
        dict: The same data structure as input.
    """
    new_maps = {}
    for age in maps:
        new_maps[age] = maps[age].copy()
        for i, sub in enumerate(maps[age]):
            indices = _method(sub, *args)
            new_maps[age][i, indices] = np.nan

    return new_maps

#---------------------
# Used in scaling data

def rescale(x, l, u):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler((l, u))
    return scaler.fit_transform(x)