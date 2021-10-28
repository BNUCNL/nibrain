import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin, Atlas, get_rois
from cxy_visual_dev.lib.algo import decompose_mf

work_dir = pjoin(proj_dir, 'analysis/decomposition')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')

    decompose_mf(
        data_files=[s1200_1096_myelin, s1200_1096_thickness],
        masks=[
            atlas.get_mask(get_rois('MMP-vis2-L'))[0],
            atlas.get_mask(get_rois('MMP-vis2-R'))[0]
        ],
        method='PCA', n_component=20, axis='subject', zscore0=None, zscore1='split',
        csv_files=[
            pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_M.csv'),
            pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_T.csv')
        ],
        cii_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'),
        pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.pkl'),
        random_state=7
    )
