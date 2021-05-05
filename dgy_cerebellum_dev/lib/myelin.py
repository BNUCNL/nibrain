import os
import tqdm
import nibabel
import numpy as np
import pandas as pd
from typing import Tuple, Callable

from . import basic, roi, preprocess

def get_vol_map(sub_id: str, subset: tuple = None) -> np.ndarray:
    """Calculates the full myelination map of given subject using 'T1w/T2w_restore.2.nii.gz'.

    Args:
        sub_id (str): Subject id.
        subset (tuple, optional): Only subset(by ROI) of myelination map will be returned if specified. Defaults to None.

    Returns:
        np.ndarray: Myelination map.
    """
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz')).get_fdata()
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz')).get_fdata()
    t1[t1 <= 0] = 0.01
    t2[t2 <= 0] = 0.01
    myelin_map = t1 / t2
    if subset:
        myelin_map = myelin_map[subset]
    
    return myelin_map

def get_vol_map_from_file(sub_id: str, subset: tuple = None, fname_template: str = '{sub_id}.nii.gz') -> np.ndarray:
    """Get full myelination map from file. '{sub_id}' in `fname_template` will be replaced by subject ID.

    Args:
        sub_id (str): Subject id.
        subset (tuple, optional): Only subset(by ROI) of myelination map will be returned if specified. Defaults to None.
        fname_template (str, optional): Template of filename. Defaults to '{sub_id}.nii.gz'.

    Returns:
        np.ndarray: Myelination map.
    """
    file = fname_template.replace('{sub_id}', sub_id)
    myelin_map = nibabel.load(file).get_fdata()
    if subset:
        myelin_map = myelin_map[subset]
    
    return myelin_map

def get_cc_map(sub_id: str) -> np.ndarray:
    """Reads myelination map of cerebral cortex of given subject from 'V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Myelination map of cerebral cortex in 1-D array.
    """
    sub_32k_dir = basic.get_32k_dir(sub_id)
    myelin_map = nibabel.load(os.path.join(sub_32k_dir, f'{sub_id}_V1_MR.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii')).get_fdata()[0]
    
    return myelin_map

def get_cb_map(sub_id: str) -> np.ndarray:
    """Wrapped version of `get_vol_map` specifying cerebellum regions.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Cerebellum myelination map.
    """
    ROI = roi.VolumeAtlas()
    roi_id = ROI.get_roi_id('CEREBELLUM')
    indices = ROI.get_idx(roi_id)
    myelin_map = get_vol_map(sub_id, subset=indices)

    return myelin_map

def get_sc_map(sub_id: str) -> np.ndarray:
    """Wrapped version of `get_vol_map` specifying subcortical regions.

    Args:
        sub_id (str): Subject ID.

    Returns:
        np.ndarray: Subcortical myelination map.
    """
    ROI = roi.VolumeAtlas()
    indices = ROI.get_sc()
    myelin_map = get_vol_map(sub_id, subset=indices)
        
    return myelin_map

def get_maps_of_all_subs(map_func: Callable, map_kargs: dict = {}, subset: Tuple = None, show_progress: bool = True) -> pd.DataFrame:
    """With specified method defined above with similar interfaces,
    applies it to all subjects and returns a pandas data frame containing data of all subjects.

    Note that in principle, this method is not limitted to myelination maps,
    and further extension to multi-modal data may require it to be extracted elsewhere.

    Args:
        map_func (Callable): Function as `get_vol_map`.
        map_kargs (dict, optional): Additional keyword arguments for `map_func`. Defaults to {}.
        subset (Tuple, optional): Further subset selection similar to `get_vol_map`. Defaults to None.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        pd.DataFrame: Pandas data frame containing data of all subjects.
    """
    maps = pd.DataFrame(columns=['sub_id', 'age', 'data'], dtype=object)
    maps['sub_id'] = basic.SUB_INFO_DF['Sub']
    maps['age'] = basic.SUB_INFO_DF['Age in years']

    n_sub = maps.shape[0]
    if show_progress:
        progress = tqdm.tqdm(total=n_sub)
        progress.set_description(f'Computing myelin maps by "{map_func.__name__}"')
    
    for i in range(n_sub):
        sub_id = maps['sub_id'][i]
        myelin_map = map_func(sub_id, **map_kargs)
        maps.at[i, 'data'] = myelin_map[subset] if subset else myelin_map
        progress.update(1) if progress else None
    
    progress.close() if progress else None
    return maps

def get_mean_maps_of_ages(maps: pd.DataFrame, niqr_outliers: float = 2.0) -> dict:
    """With data structure as output of `get_maps_of_all_subs`, calculate mean maps of each age.

    Args:
        maps (pd.DataFrame): Data of all subjects.
        niqr_outliers (float, optional): If set to `True` or equivalent values, outliers will be set to np.nan and therefore excluded. Defaults to 2.0.

    Returns:
        dict: {age: map_data}.
    """
    mean_maps = {}
    for age in basic.SUB_AGES:
        data = np.row_stack(maps[maps['age'] == age]['data'])
        if niqr_outliers:
            for i, row in enumerate(np.copy(data)):
                outliers = preprocess.niqr_outlier_indices(row, n=niqr_outliers)
                data[i, outliers] = np.nan
        
        mean_maps[age] = np.nanmean(data, axis = 0)
    
    return mean_maps