import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import LR_count_32k,\
    mmp_map_file, dataset_name2dir, dataset_name2info

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'data/HCP')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def merge_data(dataset_name, meas_name):
    """
    把所有被试的数据合并到一个cifti文件里

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
    """
    # outputs
    out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}.dscalar.nii')

    # prepare
    dataset_dir = dataset_name2dir[dataset_name]
    meas2file = {
        'myelin': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
        ),
        'thickness': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
        )
    }

    df = pd.read_csv(dataset_name2info[dataset_name])
    n_subj = df.shape[0]

    data = np.zeros((n_subj, LR_count_32k), np.float64)

    # calculate
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(sid=subj_id)
        data[subj_idx] = nib.load(meas_file).get_fdata()[0]
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')

    # save
    mmp_reader = CiftiReader(mmp_map_file)
    save2cifti(out_file, data, mmp_reader.brain_models(), df['subID'])


if __name__ == '__main__':
    merge_data(dataset_name='HCPD', meas_name='thickness')
    merge_data(dataset_name='HCPD', meas_name='myelin')
