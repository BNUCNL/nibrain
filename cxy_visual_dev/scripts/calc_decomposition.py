import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import zscore
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin, Atlas,\
    get_rois
from cxy_visual_dev.lib.algo import decompose_mf

work_dir = pjoin(proj_dir, 'analysis/decomposition')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def transform(data_files, vtx_masks, subj_mask, cat_shape, model_file,
              zscore0, csv_files):
    """
    Args:
        data_files (list): a list of file paths
            end with ".dscalar.nii"
            shape=(n_subj, LR_count_32k)
        vtx_masks (list): a list of 1D index arraies
            指定大脑区域
        subj_mask (ndarray): 1D index array
            指定使用的被试（尚未开发相关功能，目前只能设置为None）
        cat_shape (tuple): (n_row, n_col)
            按照这个形状，以行优先的顺序拼接数据
        model_file (str):
        zscore0 (str): None, split, whole
            split: do zscore across subjects of each row
            whole: do zscore across subjects of all rows
        csv_files (list): a list of CSV files
            shape=(n_subj, n_component)
            len(csv_files)=n_row
    """
    if subj_mask is not None:
        raise ValueError("subj_mask is not ready to be used.")
    n_row, n_col = cat_shape
    assert len(data_files) == n_row * n_col
    assert len(csv_files) == n_row

    n_subjects = [0]  # each row's offset and count
    data = []
    f_idx = 0
    for row_idx in range(n_row):
        data1 = []
        for col_idx in range(n_col):
            data_file = data_files[f_idx]

            # load maps
            maps = nib.load(data_file).get_fdata()

            # extract masked data
            data2 = []
            for mask in vtx_masks:
                maps_mask = maps[:, mask]
                data2.append(maps_mask)
            data2 = np.concatenate(data2, 1)

            # update
            data1.append(data2)
            f_idx += 1

        data1 = np.concatenate(data1, 1)
        if zscore0 == 'split':
            data1 = zscore(data1, 0)
        n_subjects.append(n_subjects[-1] + data1.shape[0])

        # update
        data.append(data1)
    data = np.concatenate(data, 0)

    if zscore0 == 'whole':
        data = zscore(data, 0)

    transformer = pkl.load(open(model_file, 'rb'))
    csv_data = transformer.transform(data)
    n_component = csv_data.shape[1]
    component_names = [f'C{i}' for i in range(1, n_component+1)]
    for row_idx in range(n_row):
        s_idx = n_subjects[row_idx]
        e_idx = n_subjects[row_idx + 1]
        df = pd.DataFrame(data=csv_data[s_idx:e_idx], columns=component_names)
        df.to_csv(csv_files[row_idx], index=False)


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')

    # decompose_mf(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness],
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]
    #     ], subj_mask=None, cat_shape=(2, 1), method='PCA', n_component=20,
    #     axis='subject', zscore0=None, zscore1='split',
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_T.csv')
    #     ],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.pkl'),
    #     random_state=7
    # )

    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===
    # rois = ('L_V1', 'L_V2', 'L_V3', 'R_V1', 'R_V2', 'R_V3')
    # decompose_mf(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness],
    #     vtx_masks=[atlas.get_mask(roi)[0] for roi in rois], subj_mask=None, cat_shape=(2, 1),
    #     method='PCA', n_component=20, axis='subject', zscore0=None, zscore1='split',
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj1_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj1_T.csv')
    #     ],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj1.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj1.pkl'),
    #     random_state=7
    # )
    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===

    # 左右视觉皮层拼一起，然后把myelin和thickness的数据沿顶点轴拼起来。做完zscore之后做tPCA
    # 不做跨顶点的zscore，会抹掉被试间的主要差异。
    # 做跨被试的zscore，虽然PCA第一步是为各特征减去自己的均值，但是它会记录并用在后续我要用的HCPD数据上
    # 先做跨被试的zscore可以让它记录的均值全是为0，然后transform HCPD的时候先自行对HCPD的数据做跨被试的zscore
    # decompose_mf(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness],
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]
    #     ], subj_mask=None, cat_shape=(1, 2), method='PCA', n_component=20,
    #     axis='vertex', zscore0='split', zscore1=None,
    #     csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M.dscalar.nii'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T.dscalar.nii')
    #     ],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
    #     random_state=7
    # )
    transform(
        data_files=[
            pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')
        ],
        vtx_masks=[
            atlas.get_mask(get_rois('MMP-vis2-L'))[0],
            atlas.get_mask(get_rois('MMP-vis2-R'))[0]
        ], subj_mask=None, cat_shape=(1, 2),
        model_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
        zscore0='split',
        csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_trans-HCPD.csv')]
    )

    # 把HCPD的myelin和thickness数据沿顶点轴拼起来（左右视觉皮层拼一起）。
    # 做完跨被试的zscore之后做tPCA
    # decompose_mf(
    #     fpaths=[
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')
    #     ],
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]
    #     ], subj_mask=None, cat_shape=(1, 2), method='PCA', n_component=20,
    #     axis='vertex', zscore0='split', zscore1=None,
    #     csv_files=[pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M.dscalar.nii'),
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T.dscalar.nii')
    #     ],
    #     pkl_file=pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
    #     random_state=7
    # )
