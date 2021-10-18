import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin
from cxy_visual_dev.lib.algo import pca_mf

work_dir = pjoin(proj_dir, 'analysis/PCA')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    # pca_mf(
    #     data_files=[
    #         s1200_1096_myelin, s1200_1096_thickness,
    #         pjoin(proj_dir, 'data/HCP/HCPY-alff.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPY-GBC_MMP-vis2.dscalar.nii')
    #     ],
    #     atlas_names=['MMP-vis2-LR', 'MMP-vis2-LR'],
    #     roi_names=['L_MMP_vis2', 'R_MMP_vis2'],
    #     n_component=20, axis='subject', zscore0=None, zscore1='split',
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_T.csv'),
    #         pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_A.csv'),
    #         pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_G.csv')
    #     ],
    #     cii_file=pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii'),
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.pkl'),
    #     random_state=7
    # )

    pca_mf(
        data_files=[s1200_1096_myelin, s1200_1096_thickness],
        atlas_names=['MMP-vis2-LR', 'MMP-vis2-LR'],
        roi_names=['L_MMP_vis2', 'R_MMP_vis2'],
        n_component=20, axis='subject', zscore0=None, zscore1='split',
        csv_files=[
            pjoin(work_dir, 'HCPY-M+T_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_M.csv'),
            pjoin(work_dir, 'HCPY-M+T_mask-L+R_MMP_vis2_zscore1-split_PCA-subj_T.csv')
        ],
        cii_file=pjoin(work_dir, 'HCPY-M+T_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii'),
        pkl_file=pjoin(work_dir, 'HCPY-M+T_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.pkl'),
        random_state=7
    )
