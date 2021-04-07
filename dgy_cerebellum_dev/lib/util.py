import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

from . import preprocess

def corr_func(arr1, arr2, alpha=0.05):
    _arr1, _arr2 = arr1.flatten(), arr2.flatten()
    assert len(_arr1) == len(_arr2)
    length = len(_arr1)
    
    deltas = np.arange(4 - length, length - 4 + 1)
    n_idx = len(deltas)
    results = {
        'delta': deltas,
        'r': np.ones([n_idx]),
        'sig': np.ones([n_idx]),
        'ci_lb': np.ones([n_idx]),
        'ci_ub': np.ones([n_idx])
    }
    
    for i, delta in enumerate(deltas):
        _delta = np.abs(delta)
        n = length - _delta
        if delta <= 0:
            r, sig = stats.pearsonr(_arr1[:n], _arr2[_delta:])
        else:
            r, sig = stats.pearsonr(_arr2[:n], _arr1[_delta:])
        
        zr = np.log((1 + r) / (1 - r)) / 2
        zlb = zr - stats.norm.isf(alpha / 2) / (n - 3) ** 0.5
        zub = zr + stats.norm.isf(alpha / 2) / (n - 3) ** 0.5
        results['r'][i] = r
        results['sig'][i] = sig
        results['ci_lb'][i] = (np.exp(2 * zlb) - 1) / (np.exp(2 * zlb) + 1)
        results['ci_ub'][i] = (np.exp(2 * zub) - 1) / (np.exp(2 * zub) + 1)

    return results

def plot_corr_func(corr_results, title):
    fig = plt.figure(figsize=(8, 4))
    plt.errorbar(
        corr_results['delta'], corr_results['r'],
        yerr=[corr_results['r'] - corr_results['ci_lb'], corr_results['ci_ub'] - corr_results['r']],
        fmt='o:', capsize=4, ms=5
    )
    plt.xlabel('Delta')
    plt.ylabel('r', rotation=0)
    plt.title(title)
    plt.show()

def nancorr(a: np.ndarray, b: np.ndarray) -> float:
    valid_idx = np.where(np.logical_and(np.logical_not(np.isnan(a)), np.logical_not(np.isnan(b))))
    return stats.pearsonr(a[valid_idx], b[valid_idx])

def plot_corr_mat(mean_maps, vmin = 0):
    from matplotlib import cm
    cmap = cm.viridis
    plt.figure()
    ax = plt.axes()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    matrix = np.corrcoef(np.stack(list(mean_maps.values())))
    plt.imshow(matrix - np.eye(matrix.shape[0]), cmap = cmap, vmin = vmin)
    plt.colorbar()
    plt.xticks([i for i in range(0, 17, 2)], [i + 6 for i in range(0, 17, 2)])
    plt.yticks([i for i in range(0, 17, 2)], [i + 6 for i in range(0, 17, 2)])
    plt.show()