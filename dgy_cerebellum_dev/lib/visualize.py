import numpy as np
import scipy.stats as stats
# from matplotlib import cm
# from matplotlib import pyplot as plt

from . import preprocess

# Function applied to old data structure
# def gen_corr_mat(_array, rmv_outliers = True):
#     """For a dict containing one map for each age, generate the correlation matrix.

#     Args:
#         _array (dict | np.ndarray | list): Expected to be {age: map} or [map, map, ..., map].
#         rmv_outliers (bool, optional): Whether remove outliers by default 2IQR strategy. Defaults to True.

#     Returns:
#         tuple: (corrMatrx, corrSig)
#     """
#     array = np.array(_array) if not isinstance(_array, dict) else np.array(list(_array.values()))
#     length = len(array)
    
#     normals = preprocess.get_norm_indices(array) if rmv_outliers else np.arange(array.shape[1])
#     matrix = np.zeros((length, length), dtype = np.float64)
#     sig = np.zeros((length, length), dtype = np.float64)
#     for i in range(length):
#         for j in range(length):
#             if j < i: matrix[i, j], sig[i, j] = matrix[j, i], sig[j, i]
#             else: matrix[i, j], sig[i, j] = stats.pearsonr(array[i, normals], array[j, normals])
#     return matrix, sig

# def show_corr_mat(maps, vmin = 0):
#     if isinstance(maps, dict):
#         matrix, _ = gen_corr_mat(list(maps.values()))
#     else:
#         matrix = maps
    
#     cmap = cm.viridis
#     plt.figure()
#     ax = plt.axes()
#     ax.xaxis.set_ticks_position('top')
#     ax.invert_yaxis()
#     plt.imshow(matrix - np.eye(len(maps)), cmap = cmap, vmin = vmin)
#     plt.colorbar()
#     plt.xticks([i for i in range(0, 15, 2)], [i + 8 for i in range(0, 15, 2)])
#     plt.yticks([i for i in range(0, 15, 2)], [i + 8 for i in range(0, 15, 2)])
#     plt.show()