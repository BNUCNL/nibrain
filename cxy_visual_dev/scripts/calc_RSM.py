import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k, get_rois,\
    s1200_avg_anglemirror, s1200_avg_RFsize, s1200_avg_R2, s1200_avg_curv

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'RSM')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_pearson_r_p(data1, data2, nan_mode=True):
    """
    data1的形状是m1 x n，data2的形状是m2 x n
    用data1的每一行和data2的每一行做皮尔逊相关，得到：
    m1 x m2的r矩阵和p矩阵

    如果参数nan_mode是True，则每两行做相关之前会检查并去掉值为NAN的样本点
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
    if nan_mode:
        non_nan_arr1 = ~np.isnan(data1)
        non_nan_arr2 = ~np.isnan(data2)
        for i in range(m1):
            for j in range(m2):
                non_nan_vec = np.logical_and(non_nan_arr1[i], non_nan_arr2[j])
                r, p = pearsonr(data1[i][non_nan_vec], data2[j][non_nan_vec])
                r_arr[i, j] = r
                p_arr[i, j] = p
    else:
        for i in range(m1):
            for j in range(m2):
                r, p = pearsonr(data1[i], data2[j])
                r_arr[i, j] = r
                p_arr[i, j] = p

    return r_arr, p_arr


def calc_RSM1(mask, out_file):
    """
    计算PCA的C1, C2; FA的C1, C2; DicL的C1, C2; ICA的C1, C2; distFromCalcSulc, Curvature;
    Eccentricity; PolarAngle; PolarAngleMirror; RFsize; 以及
    周明的PC1~4之间的相关矩阵。
    """
    map_PCA = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    )).get_fdata()[:2, mask]

    map_FA = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_FA-subj.dscalar.nii'
    )).get_fdata()[:2, mask]

    map_DicL = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_DicL-2-subj.dscalar.nii'
    )).get_fdata()[:, mask]
    assert map_DicL.shape[0] == 2

    map_ICA = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_ICA-2-subj.dscalar.nii'
    )).get_fdata()[:, mask]
    assert map_ICA.shape[0] == 2

    map_dist = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'
    )).get_fdata()[0, mask][None, :]

    map_curv = nib.load(s1200_avg_curv).get_fdata()[0, mask][None, :]
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_ang = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_mir = nib.load(s1200_avg_anglemirror).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_rfs = nib.load(s1200_avg_RFsize).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_zm = nib.load(pjoin(proj_dir, 'data/space/zm_PCs.dscalar.nii')).get_fdata()[:, mask]

    maps = np.concatenate([map_PCA, map_FA, map_DicL, map_ICA, map_dist, map_curv,
                           map_ecc, map_ang, map_mir, map_rfs, map_zm], 0)
    map_names = (
        'PCA-C1', 'PCA-C2', 'FA-C1', 'FA-C2', 'DicL-C1', 'DicL-C2', 'ICA-C1', 'ICA-C2',
        'distFromCalcSulc', 'Curvature', 'Eccentricity', 'Angle', 'AngleMirror', 'RFsize',
        'ZM-PC1', 'ZM-PC2', 'ZM-PC3', 'ZM-PC4'
    )

    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps)
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # >>>受阈上R2限制的HCP-MMP-visual2 mask
    atlas = Atlas('HCP-MMP')
    R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    rois = get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R')
    calc_RSM1(
        mask=np.logical_and(R2_mask, atlas.get_mask(rois)[0]),
        out_file=pjoin(work_dir, 'RSM_MMP-vis2-LR_R2.pkl')
    )
    # 受阈上R2限制的HCP-MMP-visual2 mask<<<
