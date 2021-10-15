import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from cxy_visual_dev.lib.predefine import proj_dir, Atlas

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


def MT_PC1PC2_AG():
    """
    计算HCPY的myelin，thickness平均map和
    HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii中的PC1和PC2，
    以及HCPY的ALFF，GBC平均map之间的相关矩阵。
    """

    atlas = Atlas('MMP-vis2-LR')
    roi_idx_map = atlas.maps == atlas.roi2label['L_MMP_vis2']
    map_m = nib.load(pjoin(
        anal_dir, 'mean_map/HCPY-myelin_mean.dscalar.nii')).get_fdata()[roi_idx_map][None, :]
    map_t = nib.load(pjoin(
        anal_dir, 'mean_map/HCPY-thickness_mean.dscalar.nii')).get_fdata()[roi_idx_map][None, :]
    map_PC1PC2 = nib.load(pjoin(
        anal_dir, 'PCA/HCPY-M+T+A+G_mask-L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii')).get_fdata()[:2, roi_idx_map[0]]
    map_a = nib.load(pjoin(
        anal_dir, 'mean_map/HCPY-alff_mean.dscalar.nii')).get_fdata()[roi_idx_map][None, :]
    map_g = nib.load(pjoin(
        anal_dir, 'mean_map/HCPY-GBC_MMP-vis2_mean.dscalar.nii')).get_fdata()[roi_idx_map][None, :]
    maps = np.concatenate([map_m, map_t, map_PC1PC2, map_a, map_g], 0)
    map_names = ('myelin', 'thickness', 'PC1', 'PC2', 'ALFF', 'GBC')

    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps)
    pkl.dump(data, open(pjoin(work_dir, 'MT-PC1PC2-AG_RSM.pkl'), 'wb'))


if __name__ == '__main__':
    MT_PC1PC2_AG()
