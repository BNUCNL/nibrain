import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple, Callable, Any, Dict

from . import basic, preprocess, niio, myelin, roi

def basic_grad_pca(maps: pd.DataFrame, fill_outliers=False) -> Tuple[np.ndarray, PCA]:
    """Perform gradient analysis(PCA in essence) on maps of all subjects.

    Note that `maps` is expected to be a pandas data frame with same structure as output from `get_maps_of_all_subs`.

    Args:
        maps (pd.DataFrame): Pandas data frame consisting basic information and the specific map of each subject.
        fill_outliers (bool, optional): Whether to find out outliers and fill them with feature mean before PCA. Defaults to False.

    Returns:
        Tuple[np.ndarray, PCA]: PCA object with subject maps.
    """
    matrix = np.row_stack(maps['data']) # n_sub * n_voxel
    
    if fill_outliers:
        n_outliers = []
        for i, row in enumerate(np.copy(matrix)):
            outliers = preprocess.niqr_outlier_indices(row[0])
            n_outliers.append(len(outliers))
            matrix[i, outliers] = np.nan
            feat_mean = np.nanmean(matrix[i])
            matrix[i, outliers] = feat_mean
        
        print(f'Data length: {matrix.shape[1]}')
        print('Descriptive result of n_outliers: ')
        print(stats.describe(n_outliers))
    
    matrix = matrix.T # n_voxel * n_sub
    grad_pca = PCA()
    grad_pca.fit(matrix)
    return matrix, grad_pca

def plot_pca_r2(r2: np.ndarray, roi_name=None, figsize=(7, 4), n_dot=30, n_plot=50) -> None:
    """With PCA performed, provide plots of explained variance ratio of PCA components.

    Args:
        r2 (np.ndarray): Attribute of PCA object as `explained_variance_ratio_`.
        roi_name ([type], optional): Defaults to None.
        figsize (tuple, optional): Defaults to (7, 4).
        n_dot (int, optional): Counts of components to be scattered for convenient view. Defaults to 30.
        n_plot (int, optional): Counts of components to be plotted. Defaults to 50.
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize = figsize)
    plt.scatter(np.arange(n_dot), r2[:n_dot], s = 24)
    plt.plot(np.arange(n_plot), r2[:n_plot])
    plt.ylabel('$R^2$', loc = 'center', rotation = 0, labelpad = 12)
    plt.title(f'{roi_name + " - " if roi_name else ""}Explained variance ratio of each component')
    plt.show()

def get_time_profile(grad_pca: PCA, n_comp: int = 3) -> pd.DataFrame:
    """With PCA performed, calculate `mean` and `std` of each age for each component.

    Args:
        grad_pca (PCA): PCA object.
        n_comp (int, optional): Counts of components to be analyzed. Defaults to 3.

    Returns:
        pd.DataFrame: A pandas data frame consisting time profile information.
    """
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

def plot_time_profile(time_profile: pd.DataFrame, n_comp: int, roi_name=None, figsize=(13, 4), fname=None) -> None:
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
    if fname:
        plt.savefig(fname)
    plt.show()

def get_spatial_map(matrix: np.ndarray, grad_pca: PCA, n_comp: int = 3, lbound: float = 1, ubound: float = 2) -> List[Any]:
    maps = []
    spatial_map = preprocess.rescale(grad_pca.transform(matrix), lbound, ubound).T
    for i in range(n_comp):
        maps.append(spatial_map)
    
    return maps

def grad_analysis(maps: pd.DataFrame, n_comp: int = 3, plot: bool = True, roi_name: str = None, fill_outliers: bool = False) -> Dict[str, Any]:
    """A pipeline or wrapped version of gradient analysis. Not adpoted in formal analysis.

    Args:
        maps (pd.DataFrame): [description]
        n_comp (int, optional): [description]. Defaults to 3.
        plot (bool, optional): [description]. Defaults to True.
        roi_name (str, optional): [description]. Defaults to None.
        fill_outliers (bool, optional): [description]. Defaults to False.

    Returns:
        Dict[str, Any]: Results of analysis.
    """
    matrix, grad_pca = basic_grad_pca(maps, fill_outliers)
    time_profile = get_time_profile(grad_pca, n_comp)
    if plot:
        plot_pca_r2(grad_pca.explained_variance_ratio_, roi_name)
        plot_time_profile(time_profile, n_comp, roi_name)
    spatial_map = get_spatial_map(matrix, grad_pca, n_comp)
    results =  {
        'matrix': matrix,
        'grad_pca': grad_pca,
        'time_profile': time_profile,
        'spatial_map': spatial_map
    }
    
    return results