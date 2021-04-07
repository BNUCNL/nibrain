import os
import nibabel
import numpy as np
import pandas as pd
from typing import Tuple, Callable, Union

from . import basic, roi, preprocess

def get_myelin_map(sub_id: str) -> np.ndarray:
    """Calculates the full myelination map of given subject using 'T1w/T2w_restore.2.nii.gz'.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Myelination map in 3-D array.
    """
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz')).get_fdata()
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz')).get_fdata()
    t2[t2 == 0] = 0.01
    
    myelin_map = t1 / t2
    return myelin_map

def get_cc_map(sub_id: str) -> np.ndarray:
    """Reads myelination map of cerebral cortex of given subject from 'V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Myelination map of cerebral cortex in 1-D array.
    """
    sub_32k_dir = basic.get_32k_dir(sub_id)
    myelin_map = nibabel.load(os.path.join(sub_32k_dir, f'{sub_id}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'))
    return myelin_map.get_fdata()[0], np.arange(0, 59412)

def get_cb_map(sub_id: str) -> Tuple[np.ndarray, list]:
    """Calculates myelination map of cerebellum based on 'get_myelin_map' and .2 ROI map.

    Args:
        sub_id (str): Subject ID.

    Returns:
        Tuple[np.ndarray, list]: Cerebellum myelination map in 1-D array and original indices.
    """
    myelin_map = get_myelin_map(sub_id)
    indices = roi.get_volume_indices(roi.VOLUME_ROI.CEREBELLUM)
    return myelin_map, indices

def get_sc_map(sub_id: str) -> Tuple[np.ndarray, list]:
    """Calculates myelination map of subcortical regions based on 'get_myelin_map' and .2 ROI map.

    Args:
        sub_id (str): Subject ID.

    Returns:
        Tuple[np.ndarray, list]: Subcortical myelination map in 1-D array and original indices.
    """
    myelin_map = get_myelin_map(sub_id)
    exclusion = (*roi.VOLUME_ROI.CEREBELLUM, 0)
    indices = roi.get_volume_indices(exclusion, include=False)
    return myelin_map, indices

def get_maps_of_all_subs(_map_func: Callable, step=0) -> pd.DataFrame:
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

    n_sub = maps.shape[0]
    for i in range(n_sub):
        sub_id = maps['sub_id'][i]
        myelin_map, indices = _map_func(sub_id)
        maps.at[i, 'data'] = myelin_map[indices]
        if step and (i + 1) % step == 0:
            print(f'Extracted {i+1} / {n_sub}')

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

def invert_map(data: np.ndarray, _map_func, new_indices=None, bkg_val=0) -> np.ndarray:
    assert isinstance(_map_func, Callable) and _map_func.__name__ in ['get_cb_map', 'get_sc_map']
    sub_id = basic.rand_pick_sub()
    template, indices = _map_func(sub_id)
    template[:, :, :] = bkg_val

    if not new_indices:
        new_indices = [i for i in range(data.shape[0])]
    for i, idx in enumerate(new_indices):
        template[indices[0][idx], indices[1][idx], indices[2][idx]] = data[i]

    return template