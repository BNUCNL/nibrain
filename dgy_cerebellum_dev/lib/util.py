import nibabel
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

def corr_func(arr1: np.ndarray, arr2: np.ndarray, alpha: float = 0.05) -> dict:
    """For two 1-D arrays, calculates their Pearson r sliding in different distances each time.

    For example, if `arr1` is [1, 2, 3, 4, 5] and `arr2` is [2, 3, 4, 5, 6], correlation coefficients will be calculated as:

        - delta = 0, r([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
        
        - delta = -1, r([1, 2, 3, 4], [3, 4, 5, 6])
        
        - delta = 1, r([2, 3, 4, 5], [2, 3, 4, 5])

    , where `delta` is the independent variable of correlation function.

    Note that statistical significance is also calculated, so shift stops when left array length is less than 4.
    
    Args:
        arr1 (np.ndarray): First array.
        arr2 (np.ndarray): Second array.
        alpha (float, optional): Defaults to 0.05.

    Returns:
        dict: A `dict` consisting all correlation results as 'delta', 'r', 'sig', 'ci_lb', 'ci_ub'.
    """
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

def plot_corr_func(corr_results: dict, title: str, fname: str = None):
    """Plots results given by `corr_func`.

    Args:
        corr_results (dict): Results given by `corr_func`.
        title (str): Figure title.
        fname (str, optional): Filename of output. Defaults to None.
    """
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        corr_results['delta'], corr_results['r'],
        yerr=[corr_results['r'] - corr_results['ci_lb'], corr_results['ci_ub'] - corr_results['r']],
        fmt='o:', capsize=4, ms=5
    )
    plt.xlabel('Delta')
    plt.ylabel('r', rotation=0)
    plt.title(title)
    if fname:
        plt.savefig(fname)
    plt.show()

def nancorr(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Pearson r of two 1-D arrays while ignoring `np.nan`, similar to `np.nanmean`.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        float: Correlation coefficient.
    """
    valid_idx = np.where(np.logical_and(np.logical_not(np.isnan(a)), np.logical_not(np.isnan(b))))
    return stats.pearsonr(a[valid_idx], b[valid_idx])

def plot_corr_mat(mean_maps: dict, vmin: float = 0, fname: str = None):
    """Shows correlation matrix of data as {age: mean_map}.

    Args:
        mean_maps (dict): Mean maps of all ages.
        vmin (float, optional): Minimun value of correlation matrix, set to modify color gradients. Defaults to 0.
        fname (str, optional): Output filename. Defaults to None.
    """
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
    if fname:
        plt.savefig(fname)
    plt.show()

def resample_roi_array(target_affine: np.ndarray, target_shape: tuple,
        source_affine: np.ndarray, source_shape: tuple, source_data: np.ndarray) -> np.ndarray:
    """Resamples source ROI array with affine matrices of both source space and target space.

    Since transformation is performed by matrix multiplications, target coordinate (i, j, k) is rounded to `int` type.
    It is probably acceptable for ROI data or other discrete data, but rather questionable for continuous data.
    
    Args:
        target_affine (np.ndarray): Affine matrix of target space.
        target_shape (tuple): Shape of target.
        source_affine (np.ndarray): Affine matrix of source space.
        source_shape (tuple): Shape of source.
        source_data (np.ndarray): Source data.

    Returns:
        np.ndarray: Output in target shape.
    """
    from numpy.linalg import linalg
    output_array = np.zeros(target_shape)
    output_array[:, :, :] = np.nan
    imax, jmax, kmax = target_shape
    imesh, jmesh, kmesh = np.meshgrid(np.arange(imax), np.arange(jmax), np.arange(kmax))
    imesh, jmesh, kmesh = imesh.flatten(), jmesh.flatten(), kmesh.flatten()
    coordinates = np.row_stack((imesh, jmesh, kmesh, np.ones([1, imax * jmax * kmax])))
    
    linear_transform = np.matmul(linalg.inv(source_affine), target_affine)
    new_coordinates = np.matmul(linear_transform, coordinates) # Apply transformation
    new_coordinates = np.round(new_coordinates[:3]).astype(int) # Remove axis 3, retaining axis 0, 1, and 2

    # Remove indices lower than 0 or exceeding original data shape on any axis
    # A legal 3-D coordinate should satisfy following requirements:
    # 1. Not lower than 0;
    # 2. Not exceeding shape of source data
    legal = np.where(np.sum((new_coordinates >= 0) * (new_coordinates < np.array([source_shape]).T), axis=0) == 3) # `legal` is a tuple
    output_array[(imesh[legal], jmesh[legal], kmesh[legal])] = source_data[tuple(new_coordinates[:, legal[0]])]

    return output_array

def resample_roi_map(ftarget: str, fsource: str, fname: str) -> nibabel.nifti1.Nifti1Image:
    """Wrapped version of `resample_roi_array`.

    Args:
        ftarget (str): Target image file.
        fsource (str): Source image file(ROI map).
        fname (str): Output filename.

    Returns:
        nibabel.nifti1.Nifti1Image: Resampled `nibabel` image object.
    """
    target, source = nibabel.load(ftarget), nibabel.load(fsource)
    array = resample_roi_array(target.affine, target.shape, source.affine, source.shape, source.get_fdata())
    img = nibabel.nifti1.Nifti1Image(array, target.affine)
    nibabel.save(img, fname)
    return img