import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore
from os.path import join as pjoin
from sklearn.decomposition import PCA
from cxy_visual_dev.lib.predefine import Atlas, L_offset_32k, L_count_32k,\
    R_offset_32k, R_count_32k, LR_count_32k

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/structure')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_TM(dataset_name='HCPD', meas_name='thickness', atlas_name='LR',
            zscore_flag=False):
    """
    Calculate thickness or myelination
    """
    # inputs
    meas_file = pjoin(proj_dir,
                      f'data/HCP/{dataset_name}_{meas_name}.dscalar.nii')

    # outputs
    if zscore_flag:
        out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}_'
                                   f'{atlas_name}_zscore.csv')
    else:
        out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}_'
                                   f'{atlas_name}.csv')

    # prepare
    meas_maps = nib.load(meas_file).get_fdata()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    out_df = pd.DataFrame()

    # calculate
    if zscore_flag:
        meas_maps_L = meas_maps[:, L_offset_32k:(L_offset_32k+L_count_32k)]
        meas_maps_R = meas_maps[:, R_offset_32k:(R_offset_32k+R_count_32k)]
        meas_maps_L = zscore(meas_maps_L, 1)
        meas_maps_R = zscore(meas_maps_R, 1)
        meas_maps[:, L_offset_32k:(L_offset_32k+L_count_32k)] = meas_maps_L
        meas_maps[:, R_offset_32k:(R_offset_32k+R_count_32k)] = meas_maps_R
        del meas_maps_L, meas_maps_R

    for roi, lbl in atlas.roi2label.items():
        meas_vec = np.mean(meas_maps[:, atlas.maps[0] == lbl], 1)
        out_df[roi] = meas_vec

    # save
    out_df.to_csv(out_file, index=False)


def pca(fpath, atlas_name, roi_name, n_component, out_csv, out_pkl):
    # prepare
    component_names = [f'C{i}' for i in range(1, n_component+1)]

    meas_maps = nib.load(fpath).get_fdata()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    meas_maps = meas_maps[:, roi_idx_map]

    # calculate
    pca = PCA(n_components=n_component)
    pca.fit(meas_maps)
    meas_components = pca.transform(meas_maps)

    # save
    out_df = pd.DataFrame(data=meas_components, columns=component_names)
    out_df.to_csv(out_csv, index=False)
    pkl.dump(pca, open(out_pkl, 'wb'))


if __name__ == '__main__':
    # HCP_MMP1 atlas 包含 Cole_visual_ROI
    # calc_TM(dataset_name='HCPD', meas_name='myelin', atlas_name='HCP_MMP1')
    # calc_TM(dataset_name='HCPD', meas_name='thickness', atlas_name='HCP_MMP1')
    # calc_TM(dataset_name='HCPD', meas_name='myelin', atlas_name='LR')
    # calc_TM(dataset_name='HCPD', meas_name='thickness', atlas_name='LR')
    # calc_TM(dataset_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_LR')
    # calc_TM(dataset_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_LR')
    # calc_TM(dataset_name='HCPD', meas_name='myelin', atlas_name='HCP_MMP1', zscore_flag=True)
    # calc_TM(dataset_name='HCPD', meas_name='thickness', atlas_name='HCP_MMP1', zscore_flag=True)
    pca(
        fpath=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
        atlas_name='Cole_visual_LR', roi_name='R_cole_visual', n_component=20,
        out_csv=pjoin(work_dir, 'HCPD_thickness_4mm_PCA-20.csv'),
        out_pkl=pjoin(work_dir, 'HCPD_thickness_4mm_PCA-20.pkl')
    )
    pca(
        fpath=pjoin(proj_dir, 'data/HCP/HCPD_myelin_4mm.dscalar.nii'),
        atlas_name='Cole_visual_LR', roi_name='R_cole_visual', n_component=20,
        out_csv=pjoin(work_dir, 'HCPD_myelin_4mm_PCA-20.csv'),
        out_pkl=pjoin(work_dir, 'HCPD_myelin_4mm_PCA-20.pkl')
    )
