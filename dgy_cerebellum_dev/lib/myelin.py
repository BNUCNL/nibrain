import os
import tqdm
import pickle
import nibabel
import numpy as np
import pandas as pd
from typing import Tuple, Callable, Union

from . import basic, roi, preprocess

SET_CACHE = False
USE_CACHE = False
def LOCAL_CACHE_DIR():
    return os.path.join(basic.CACHE_DIR(), 'myelin')

def get_lu_bounds(l, u, u_max, semi_width):
    if l < 0:
        return 0, 2 * semi_width + 1
    elif u > u_max:
        return u_max - 2 * semi_width - 1, u_max
    else:
        return l, u

def get_vol_myelin_map(sub_id: str, fill_outliers_strategy='neighbor', n_neighbors=1, n_iqr=2.0) -> np.ndarray:
    """Calculates the full myelination map of given subject using 'T1w/T2w_restore.2.nii.gz'.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Myelination map in 3-D array.
    """
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz')).get_fdata()
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz')).get_fdata()
    t2[t2 <= 0] = 0.01
    myelin_map = t1 / t2
    
    if fill_outliers_strategy == 'neighbor':
        outliers = preprocess.niqr_outlier_indices(myelin_map, n_iqr)
        myelin_map[outliers] = np.nan
        for x, y, z in zip(*outliers):
            xl, xu = get_lu_bounds(x - n_neighbors, x + n_neighbors + 1, myelin_map.shape[0], n_neighbors)
            yl, yu = get_lu_bounds(y - n_neighbors, y + n_neighbors + 1, myelin_map.shape[1], n_neighbors)
            zl, zu = get_lu_bounds(z - n_neighbors, z + n_neighbors + 1, myelin_map.shape[2], n_neighbors)
            new_val = np.nanmean(myelin_map[xl: xu, yl: yu, zl: zu])
            myelin_map[x, y, z] = new_val
    elif fill_outliers_strategy == 'winsorize':
        l, u = preprocess.get_iqr(myelin_map)
        myelin_map[myelin_map < l] = l
        myelin_map[myelin_map > u] = u

    if SET_CACHE:
        with open(os.path.join(LOCAL_CACHE_DIR(), sub_id), 'wb') as f:
            pickle.dump(myelin_map, f)
    
    return myelin_map

def get_cc_map(sub_id: str, fill_outliers='neighbor', n_neighbors=2, n_iqr=2.0) -> np.ndarray:
    """Reads myelination map of cerebral cortex of given subject from 'V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'.

    Args:
        sub_id (str): Subject ID.
        fill_outliers (bool): Whether fill outliers with `fill_outliers_with_surf_neighbors`.

    Returns:
        np.ndarray: Myelination map of cerebral cortex in 1-D array.
    """
    sub_32k_dir = basic.get_32k_dir(sub_id)
    myelin_map = nibabel.load(os.path.join(sub_32k_dir, f'{sub_id}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii')).get_fdata()[0]
    if fill_outliers:
        outliers = preprocess.niqr_outlier_indices(myelin_map, n_iqr)
        myelin_map[outliers] = np.nan
        for outidx in outliers[0]:
            myelin_map[outidx] = np.nanmean(myelin_map[max(0, outidx - n_neighbors): outidx + n_neighbors + 1])
    
    return myelin_map, np.arange(0, 59412)

def get_cb_map(sub_id: str, fill_outliers='neighbor', n_neighbors=1, n_iqr=2.0) -> Tuple[np.ndarray, list]:
    """Calculates myelination map of cerebellum based on 'get_myelin_map' and .2 ROI map.

    Args:
        sub_id (str): Subject ID.
        fill_outliers (bool): Whether fill outliers with `fill_outliers_with_vol_neighbors`.

    Returns:
        Tuple[np.ndarray, list]: Cerebellum myelination map in 1-D array and original indices.
    """
    if USE_CACHE and os.path.isfile(os.path.join(LOCAL_CACHE_DIR(), sub_id)):
        with open(os.path.join(LOCAL_CACHE_DIR(), sub_id), 'rb') as f:
            myelin_map = pickle.load(f)
    else:
        myelin_map = get_vol_myelin_map(sub_id, fill_outliers, n_neighbors, n_iqr)
    
    indices = roi.get_volume_indices(roi.VOLUME_ROI.CEREBELLUM)
    # if fill_outliers:
    #     myelin_map = fill_outliers_with_vol_neighbors(myelin_map, indices)

    return myelin_map, indices

def get_sc_map(sub_id: str, fill_outliers='neighbor', n_neighbors=1, n_iqr=2.0) -> Tuple[np.ndarray, list]:
    """Calculates myelination map of subcortical regions based on 'get_myelin_map' and .2 ROI map.

    Args:
        sub_id (str): Subject ID.
        fill_outliers (bool): Whether fill outliers with `fill_outliers_with_vol_neighbors`.

    Returns:
        Tuple[np.ndarray, list]: Subcortical myelination map in 1-D array and original indices.
    """
    if USE_CACHE and os.path.isfile(os.path.join(LOCAL_CACHE_DIR(), sub_id)):
        with open(os.path.join(LOCAL_CACHE_DIR(), sub_id), 'rb') as f:
            myelin_map = pickle.load(f)
    else:
        myelin_map = get_vol_myelin_map(sub_id, fill_outliers, n_neighbors, n_iqr)
    
    exclusion = (*roi.VOLUME_ROI.CEREBELLUM, 0)
    indices = roi.get_volume_indices(exclusion, include=False)
    # if fill_outliers:
    #     myelin_map = fill_outliers_with_vol_neighbors(myelin_map, indices)
        
    return myelin_map, indices

def get_maps_of_all_subs(_map_func: Callable, show_progress=False) -> pd.DataFrame:
    """Returns a pandas dataframe containing subject data and myelination maps of required ROI.

    Args:
        _map_func (Callable): Method to designate ROI.
        step (int, optional): Step for progress informing. Defaults to 0.

    Returns:
        pd.DataFrame: Dataframe with 3 columns: 'sub_id', 'age', and 'data'. Whole array of myelination map is stored in one unit.
    """
    assert callable(_map_func) and _map_func.__name__ in ['get_cc_map', 'get_cb_map', 'get_sc_map']
    maps = pd.DataFrame(columns=['sub_id', 'age', 'data'], dtype=object)
    maps['sub_id'] = basic.SUB_INFO_DF['Sub']
    maps['age'] = basic.SUB_INFO_DF['Age in years']
    if SET_CACHE and not os.path.isdir(LOCAL_CACHE_DIR()):
        os.mkdir(LOCAL_CACHE_DIR())

    n_sub = maps.shape[0]
    if show_progress:
        progress = tqdm.tqdm(total=n_sub)
        progress.set_description(f'Computing myelin maps by "{_map_func.__name__}"')
    
    for i in range(n_sub):
        sub_id = maps['sub_id'][i]
        myelin_map, indices = _map_func(sub_id)
        maps.at[i, 'data'] = myelin_map[indices]
        progress.update(1) if progress else None
    
    progress.close() if progress else None
    return maps

def get_mean_maps_of_ages(maps: pd.DataFrame, remove_outliers=True) -> dict:
    mean_maps = {}
    for age in basic.SUB_AGES:
        data = np.row_stack(maps[maps['age'] == age]['data'])
        if remove_outliers:
            for i, row in enumerate(np.copy(data)):
                outliers = preprocess.niqr_outlier_indices(row)
                data[i, outliers] = np.nan
            mean_maps[age] = np.nanmean(data, axis = 0)
        else:
            mean_maps[age] = np.mean(data, axis = 0)
    
    return mean_maps

def invert_map(data: np.ndarray, _map_func, new_indices=None, bkg_val=np.nan) -> np.ndarray:
    assert isinstance(_map_func, Callable) and _map_func.__name__ in ['get_cb_map', 'get_sc_map']
    sub_id = basic.rand_pick_sub()
    template, indices = _map_func(sub_id, fill_outliers=None)
    template[:, :, :] = bkg_val

    if not new_indices:
        new_indices = [i for i in range(data.shape[0])]
    for i, idx in enumerate(new_indices):
        template[indices[0][idx], indices[1][idx], indices[2][idx]] = data[i]

    return template