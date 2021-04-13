import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple, Callable, Any, Dict

from . import basic, preprocess, niio, myelin

def basic_grad_pca(maps: pd.DataFrame, fill_outliers_with_mean=False) -> Tuple[np.ndarray, PCA]:
    from sklearn.decomposition import PCA
    matrix = np.row_stack(maps['data']) # n_sub * n_voxel
    
    if fill_outliers_with_mean:
        n_outliers = []
        for i, row in enumerate(np.copy(matrix)):
            outliers = preprocess.niqr_outlier_indices(row[0])
            n_outliers.append(len(outliers))
            matrix[i, outliers] = np.nan
            feat_mean = np.nanmean(matrix[i])
            matrix[i, outliers] = feat_mean
        
        print(f'n_voxel: {matrix.shape[1]}')
        print('Descriptive result of n_outliers: ')
        print(stats.describe(n_outliers))
    
    matrix = matrix.T # n_voxel * n_sub
    grad_pca = PCA()
    grad_pca.fit(matrix)
    return matrix, grad_pca

def plot_pca_r2(r2: np.ndarray, roi_name=None, figsize=(7, 4), n_dot=30, n_plot=50) -> None:
    plt.style.use('ggplot')
    plt.figure(figsize = figsize)
    plt.scatter(np.arange(n_dot), r2[:n_dot], s = 24)
    plt.plot(np.arange(n_plot), r2[:n_plot])
    plt.ylabel('$R^2$', loc = 'center', rotation = 0, labelpad = 12)
    plt.title(f'{roi_name + " - " if roi_name else ""}Explained variance ratio of each component')
    plt.show()

def get_time_profile(grad_pca: PCA, n_comp: int = 3) -> pd.DataFrame:
    w_sub = pd.DataFrame(columns=['age', 'data'])
    w_sub['age'] = basic.SUB_INFO_DF['Age in years']
    w_sub['data'] = list(grad_pca.components_[:n_comp, :].T)

    time_profile = pd.DataFrame(columns=['age', 'mean', 'se'])
    time_profile['age'] = basic.SUB_AGES
    for i, age in enumerate(basic.SUB_AGES):
        w_group = np.row_stack(w_sub[w_sub['age'] == age]['data'])
        time_profile.at[i, 'mean'] = w_group.mean(axis=0)
        time_profile.at[i, 'se'] = w_group.std(axis=0, ddof=1) / w_group.shape[0] ** 0.5 # n_sub
    
    return time_profile

def plot_time_profile(time_profile: pd.DataFrame, n_comp: int, roi_name=None, figsize=(13, 4)) -> None:
    w_mean = np.row_stack(time_profile['mean'])
    w_se = np.row_stack(time_profile['se'])

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, n_comp, figsize = figsize)
    fig.suptitle(f'{roi_name + " - " if roi_name else ""}Components development', fontsize = 20)
    cmap = ['#e2705f', '#6ca6c9', '#afa7dc']
    for i in range(n_comp):
        axes[i].errorbar(
            basic.SUB_AGES, w_mean[:, i], w_se[:, i],
            capsize = 4, color = cmap[i % 3]
        )
        axes[i].set_title(f'PC{i+1}', y = -0.2)
        axes[i].set_xticks(np.arange(6, 24, 2))
    
    plt.tight_layout()
    plt.show()

def get_spatial_map(matrix: np.ndarray, grad_pca: PCA, nib_type: str, _map_func: Callable = None, n_comp: int = 3) -> List[Any]:
    nib_objects = []
    spatial_map = preprocess.rescale(grad_pca.transform(matrix), 1, 2).T
    for i in range(n_comp):
        if nib_type == 'nifti' and _map_func:
            nib_data = myelin.invert_map(spatial_map[i], _map_func)
            nib_objects.append(niio.get_volume_obj(nib_data))
        elif nib_type == 'cifti':
            nib_data = np.reshape(spatial_map[i], (1, 59412))
            nib_objects.append(niio.get_surface_obj(nib_data))
        else:
            raise ValueError('Invalid nibabel type.')
    
    return nib_objects

def grad_analysis(maps: pd.DataFrame, nib_type: str, _map_func: Callable = None, n_comp: int = 3, plot: bool = True, roi_name: str = None, fill_outliers: bool = False) -> Dict[str, Any]:
    matrix, grad_pca = basic_grad_pca(maps, fill_outliers)
    time_profile = get_time_profile(grad_pca, n_comp)
    if plot:
        plot_pca_r2(grad_pca.explained_variance_ratio_, roi_name)
        plot_time_profile(time_profile, n_comp, roi_name)
    spatial_map = get_spatial_map(matrix, grad_pca, nib_type, _map_func, n_comp)
    results =  {
        'matrix': matrix,
        'grad_pca': grad_pca,
        'time_profile': time_profile,
        'spatial_map': spatial_map
    }
    
    return results