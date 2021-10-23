import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k, get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'RSM')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_pearson_r_p(data1, data2):
    """
    data1的形状是m1 x n，data2的形状是m2 x n
    用data1的每一行和data2的每一行做皮尔逊相关，得到：
    m1 x m2的r矩阵和p矩阵
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    assert data1.ndim == 2
    assert data2.ndim == 2
    m1, n = data1.shape
    m2, n2 = data2.shape
    assert n == n2

    r_arr = np.zeros((m1, m2), np.float64)
    p_arr = np.zeros((m1, m2), np.float64)
    for i in range(m1):
        for j in range(m2):
            r, p = pearsonr(data1[i], data2[j])
            r_arr[i, j] = r
            p_arr[i, j] = p

    return r_arr, p_arr


def EA_C2(mask, out_file):
    """
    计算HCPY的eccentricity，polar angle平均map和
    HCPY-M+T_L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii中的C2，以及
    HCPY-M+T_L+R_MMP_vis2_zscore1-split_FA-subj.dscalar.nii中的C2
    之间的相关矩阵。
    """
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k][mask]
    map_ang = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k][mask]
    map_PCA = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii')).get_fdata()[1, mask]
    map_FA = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_L+R_MMP_vis2_zscore1-split_FA-subj.dscalar.nii')).get_fdata()[1, mask]

    nan_vec = np.zeros_like(map_ecc, bool)
    for i in (map_ecc, map_ang, map_PCA, map_FA):
        nan_vec = np.logical_or(nan_vec, np.isnan(i))
    if np.all(nan_vec):
        raise ValueError
    non_nan_vec = ~nan_vec

    map_ecc = map_ecc[non_nan_vec][None, :]
    map_ang = map_ang[non_nan_vec][None, :]
    map_PCA = map_PCA[non_nan_vec][None, :]
    map_FA = map_FA[non_nan_vec][None, :]
    maps = np.concatenate([map_ecc, map_ang, map_PCA, map_FA], 0)
    map_names = ('eccentricity', 'angle', 'PCA-C2', 'FA-C2')

    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps)
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # EA_C2(
    #     mask=Atlas('Wang2015').get_mask(['L'])[0],
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_Wang2015-L.pkl')
    # )
    # EA_C2(
    #     mask=Atlas('Wang2015').get_mask(['R'])[0],
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_Wang2015-R.pkl')
    # )
    # EA_C2(
    #     mask=Atlas('Wang2015').get_mask(['LR'])[0],
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_Wang2015-LR.pkl')
    # )

    # >>>受阈上R2限制的HCP-MMP-visual2 mask
    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(
    #     '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'
    #     'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'
    #     'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'
    # ).get_fdata()[0, :LR_count_32k] > 9.8

    # rois = get_rois('MMP-vis2-L')
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-L_R2.pkl')
    # )

    # rois = get_rois('MMP-vis2-R')
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-R_R2.pkl')
    # )

    # rois = get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R')
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-LR_R2.pkl')
    # )
    # 受阈上R2限制的HCP-MMP-visual2 mask<<<

    # >>>受阈上R2限制的HCP-MMP-visual2 早期及其它视觉mask
    # rois_early_L = ['L_V1', 'L_V2', 'L_V3', 'L_V4']
    # rois_early_R = ['R_V1', 'R_V2', 'R_V3', 'R_V4']
    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(
    #     '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'
    #     'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'
    #     'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'
    # ).get_fdata()[0, :LR_count_32k] > 9.8

    # rois = rois_early_L
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early-L_R2.pkl')
    # )

    # rois = rois_early_R
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early-R_R2.pkl')
    # )

    # rois = rois_early_L + rois_early_R
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early-LR_R2.pkl')
    # )
    # 受阈上R2限制的HCP-MMP-visual2 mask<<<

    # >>>受阈上R2限制的HCP-MMP-visual2 早期及其它视觉mask2
    # rois_early_L = ['L_V1', 'L_V2', 'L_V3']
    # rois_early_R = ['R_V1', 'R_V2', 'R_V3']
    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(
    #     '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'
    #     'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'
    #     'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'
    # ).get_fdata()[0, :LR_count_32k] > 9.8

    # rois = rois_early_L
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early2-L_R2.pkl')
    # )

    # rois = rois_early_R
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early2-R_R2.pkl')
    # )

    # rois = rois_early_L + rois_early_R
    # EA_C2(
    #     mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
    #     out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early2-LR_R2.pkl')
    # )
    # 受阈上R2限制的HCP-MMP-visual2 mask2<<<

    # >>>受阈上R2限制的HCP-MMP-visual2 早期及其它视觉mask3
    rois_early_L = ['L_V1', 'L_V2']
    rois_early_R = ['R_V1', 'R_V2']
    atlas = Atlas('HCP-MMP')
    R2_mask = nib.load(
        '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'
        'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'
        'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'
    ).get_fdata()[0, :LR_count_32k] > 9.8

    rois = rois_early_L
    EA_C2(
        mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
        out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early3-L_R2.pkl')
    )

    rois = rois_early_R
    EA_C2(
        mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
        out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early3-R_R2.pkl')
    )

    rois = rois_early_L + rois_early_R
    EA_C2(
        mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
        out_file=pjoin(work_dir, 'EA_C2_RSM_MMP-vis2-early3-LR_R2.pkl')
    )
    # 受阈上R2限制的HCP-MMP-visual2 mask2<<<
