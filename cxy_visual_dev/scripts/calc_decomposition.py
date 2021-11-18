import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin, Atlas,\
    get_rois
from cxy_visual_dev.lib.algo import decompose, transform

work_dir = pjoin(proj_dir, 'analysis/decomposition')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')

    # 在成人数据上，把左右HCP-MMP_visual-cortex2做zscore之后拼一起
    # 联合myelin和thickness做空间PCA
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
    #     method='PCA', axis=0,
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_T.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0=None, zscore1='split', n_component=20, random_state=7
    # )

    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===
    # rois = ('L_V1', 'L_V2', 'L_V3', 'R_V1', 'R_V2', 'R_V3')
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
    #     method='PCA', axis=0,
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj_T.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj.pkl'),
    #     vtx_masks=[atlas.get_mask(roi)[0] for roi in rois], map_mask=None,
    #     zscore0=None, zscore1='split', n_component=20, random_state=7
    # )
    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===

    # 左右视觉皮层拼一起，然后把myelin和thickness的数据沿顶点轴拼起来。做完zscore之后做tPCA
    # 不做跨顶点的zscore，会抹掉被试间的主要差异。
    # 做跨被试的zscore，虽然PCA第一步是为各特征减去自己的均值，但是它会记录并用在后续我要用的HCPD数据上
    # 先做跨被试的zscore可以让它记录的均值全是为0，然后transform HCPD的时候先自行对HCPD的数据做跨被试的zscore
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(1, 2),
    #     method='PCA', axis=1,
    #     csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_new.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M_new.dscalar.nii'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T_new.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_new.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split', zscore1=None, n_component=20, random_state=7
    # )
    # transform(
    #     fpaths=[
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')],
    #     cat_shape=(1, 2),
    #     model_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_new.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split',
    #     csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_trans-HCPD_new.csv')]
    # )

    # 把HCPD的myelin和thickness数据沿顶点轴拼起来（左右视觉皮层拼一起）。
    # 做完跨被试的zscore之后做tPCA
    # decompose(
    #     fpaths=[
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')],
    #     cat_shape=(1, 2),
    #     method='PCA', axis=1,
    #     csv_files=[pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_new.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M_new.dscalar.nii'),
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T_new.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_new.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split', zscore1=None, n_component=20, random_state=7
    # )

    # 在成人数据上，对右脑HCP-MMP1_visual-cortex3做zscore
    # 联合myelin和thickness做空间PCA
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
    #     method='PCA', axis=0,
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_T.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.pkl'),
    #     vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]],
    #     map_mask=None, zscore0=None, zscore1='split', n_component=20, random_state=7
    # )

    # 在成人数据上，联合myelin和thickness做右脑HCP-MMP1_visual-cortex3的空间PCA
    decompose(
        fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
        method='PCA', axis=0,
        csv_files=[
            pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_PCA-subj_M.csv'),
            pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_PCA-subj_T.csv')],
        cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_PCA-subj.dscalar.nii')],
        pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_PCA-subj.pkl'),
        vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]],
        map_mask=None, zscore0=None, zscore1=None, n_component=20, random_state=7
    )
