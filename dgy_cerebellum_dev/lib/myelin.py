import os
import nibabel
import numpy as np
import pandas as pd

from . import basic, roi

def get_myelin_map(sub_id):
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz'))
    return t1.get_fdata() / t2.get_fdata()

def get_cc_map(sub_id):
    sub_32k_dir = basic.get_32k_dir(sub_id)
    myelin_map = nibabel.load(os.path.join(sub_32k_dir, f'{sub_id}_V1_MR.MyelinMap.32k_fs_LR.dscalar.nii'))
    return myelin_map.get_fdata()[0]

def get_cb_map(sub_id):
    myelin_map = get_myelin_map(sub_id)
    indices = roi.get_volume_indices(roi.VOLUME_ROI.CEREBELLUM)
    return myelin_map, indices

def get_sc_map(sub_id):
    myelin_map = get_myelin_map(sub_id)
    exclusion = (*roi.VOLUME_ROI.CEREBELLUM, 0)
    indices = roi.get_volume_indices(exclusion, include=False)
    return myelin_map, indices

def get_maps_of_all_subs(_map_func, step=20):
    assert callable(_map_func) and _map_func.__name__ in ['get_cb_map', 'get_sc_map']
    maps = pd.DataFrame(columns=['sub_id', 'age', 'data'])
    maps['sub_id'] = basic.SUB_INFO_DF['Sub']
    maps['age'] = basic.SUB_INFO_DF['Age in years']

    n_sub = maps.shape[0]
    for i in range(n_sub):
        sub_id = maps['sub_id'][i]
        myelin_map, indices = _map_func(sub_id)
        maps['data'][i] = myelin_map[indices]
        if i % step == 0:
            print(f'Extracted {i+1} / {n_sub}')

    return maps

def get_mean_maps_of_ages(maps: pd.DataFrame, _mean_func=np.nanmean):
    mean_maps = {}
    for age in basic.SUB_AGES:
        data = np.row_stack(maps[maps['age'] == age]['data'])
        mean_maps[age] = _mean_func(data, axis = 0)
    return mean_maps

def invert_map(data: np.ndarray, _map_func, new_indices=None, bkg_val=0):
    assert callable(_map_func) and _map_func.__name__ in ['get_cb_map', 'get_sc_map']
    sub_id = basic.rand_pick_sub()
    template, indices = _map_func(sub_id)
    template[:, :, :] = bkg_val

    if not new_indices:
        new_indices = [i for i in range(data.shape[0])]
    for i, idx in enumerate(new_indices):
        template[indices[0][idx], indices[1][idx], indices[2][idx]] = data[i]

    return template