import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from magicbox.io.io import CiftiReader
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k, get_rois,\
    s1200_avg_RFsize, s1200_avg_R2, s1200_avg_curv
from cxy_visual_dev.lib.algo import cat_data_from_cifti

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'RSM')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_pearson_r_p(data1, data2, nan_mode=False):
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


def calc_RSM1(src_file, mask, out_file):
    """
    计算PCA的C1, C2; distFromCalcSulc; Curvature; VertexArea;
    Eccentricity; PolarAngle; RFsize; 以及
    周明的PC1~4之间的相关矩阵。
    """
    map_PCA = nib.load(src_file).get_fdata()[:2, mask]

    map_dist = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'
    )).get_fdata()[0, mask][None, :]

    reader = CiftiReader(s1200_avg_curv)
    curv_l, _, idx2v_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT')
    curv_r, _, idx2v_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT')
    map_curv = np.c_[curv_l, curv_r][0, mask][None, :]

    va_l = nib.load('/nfs/p1/public_dataset/datasets/hcp/DATA/'
                    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                    'S1200.L.midthickness_MSMAll_va.32k_fs_LR.shape.gii').darrays[0].data
    va_r = nib.load('/nfs/p1/public_dataset/datasets/hcp/DATA/'
                    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                    'S1200.R.midthickness_MSMAll_va.32k_fs_LR.shape.gii').darrays[0].data
    map_va = np.r_[va_l[idx2v_l], va_r[idx2v_r]][mask][None, :]

    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_ang = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_rfs = nib.load(s1200_avg_RFsize).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_zm = nib.load(pjoin(proj_dir, 'data/space/zm_PCs.dscalar.nii')).get_fdata()[:, mask]

    maps = np.concatenate([map_PCA, map_dist, map_curv, map_va,
                           map_ecc, map_ang, map_rfs, map_zm], 0)
    map_names = (
        'PCA-C1', 'PCA-C2', 'distFromOP', 'Curvature', 'VertexArea',
        'Eccentricity', 'Angle', 'RFsize', 'ZM-PC1', 'ZM-PC2', 'ZM-PC3', 'ZM-PC4')

    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps, True)
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM2():
    """
    计算各年龄内被试之间thickness或myelin的空间pattern的相似性矩阵
    做半脑的时候不用zscore，因为皮尔逊相关本来就是要减均值和除标准差的。
    """
    # prepare visual cortex mask
    atlas = Atlas('HCP-MMP')
    masks = [
        atlas.get_mask(get_rois('MMP-vis3-R'))[0]
    ]

    # prepare sptial pattern
    meas_name = 'thickness'
    data_file = pjoin(proj_dir, f'data/HCP/HCPD_{meas_name}.dscalar.nii')
    data = cat_data_from_cifti([data_file], (1, 1), masks, zscore1=None)[0]

    # prepare ages
    info_df = pd.read_csv(dataset_name2info['HCPD'])
    ages = np.array(info_df['age in years'])
    ages_uniq = np.unique(ages)

    # calculating
    out_file = pjoin(work_dir, 'RSM_HCPD-{0}_MMP-vis3-R_age-{1}.pkl')
    for age in ages_uniq:
        idx_vec = ages == age
        names = info_df.loc[idx_vec, 'subID'].to_list()
        data_tmp = data[idx_vec]
        out_dict = {'row_name': names, 'col_name': names}
        out_dict['r'], out_dict['p'] = calc_pearson_r_p(data_tmp, data_tmp, False)
        pkl.dump(out_dict, open(out_file.format(meas_name, age), 'wb'))


if __name__ == '__main__':
    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    # rois_L = get_rois('MMP-vis2-L')
    # rois_R = get_rois('MMP-vis2-R')
    # rois_LR = rois_L + rois_R

    # >>>MMP-vis3-R mask
    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    # mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]
    # src_file = pjoin(
    #     anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    # )
    # calc_RSM1(
    #     src_file=src_file, mask=mask,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis3-R.pkl')
    # )
    # calc_RSM1(
    #     src_file=src_file, mask=np.logical_and(R2_mask, mask),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis3-R_R2.pkl')
    # )
    # MMP-vis3-R mask<<<

    # >>>HCP-MMP-visual2早期及其它视觉mask
    # src_file = pjoin(
    #     anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    # )
    # rois_early = get_rois('MMP-vis2-G1') + get_rois('MMP-vis2-G2')
    # print('rois_early:', rois_early)
    # rois_early_L = [f'L_{roi}' for roi in rois_early]
    # rois_early_R = [f'R_{roi}' for roi in rois_early]
    # rois_early_LR = rois_early_L + rois_early_R

    # mask_early_LR = atlas.get_mask(rois_early_LR)[0]
    # calc_RSM1(
    #     mask=mask_early_LR,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-early-LR.pkl')
    # )
    # calc_RSM1(
    #     mask=np.logical_and(R2_mask, mask_early_LR),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-early-LR_R2.pkl')
    # )

    # rois_later_LR = rois_LR.copy()
    # for roi in rois_early_LR:
    #     rois_later_LR.remove(roi)
    # mask_later_LR = atlas.get_mask(rois_later_LR)[0]
    # calc_RSM1(
    #     mask=mask_later_LR,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-later-LR.pkl')
    # )
    # calc_RSM1(
    #     mask=np.logical_and(R2_mask, mask_later_LR),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-later-LR_R2.pkl')
    # )
    # HCP-MMP-visual2早期及其它视觉mask<<<

    # >>>HCP-MMP-visual2 3+16+17+18 groups (dorsal)
    # rois_dorsal = get_rois('MMP-vis2-G3') + get_rois('MMP-vis2-G16') +\
    #     get_rois('MMP-vis2-G17') + get_rois('MMP-vis2-G18')
    # print('rois_dorsal:', rois_dorsal)
    # rois_dorsal_L = [f'L_{roi}' for roi in rois_dorsal]
    # rois_dorsal_R = [f'R_{roi}' for roi in rois_dorsal]
    # rois_dorsal_LR = rois_dorsal_L + rois_dorsal_R

    # mask_dorsal_LR = atlas.get_mask(rois_dorsal_LR)[0]
    # calc_RSM1(
    #     mask=mask_dorsal_LR,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-dorsal-LR.pkl')
    # )
    # calc_RSM1(
    #     mask=np.logical_and(R2_mask, mask_dorsal_LR),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-dorsal-LR_R2.pkl')
    # )
    # # HCP-MMP-visual2 3+16+17+18 groups (dorsal)<<<

    # # >>>HCP-MMP-visual2 4+13+14 groups (ventral)
    # rois_ventral = get_rois('MMP-vis2-G4') + get_rois('MMP-vis2-G13') +\
    #     get_rois('MMP-vis2-G14')
    # print('rois_ventral:', rois_ventral)
    # rois_ventral_L = [f'L_{roi}' for roi in rois_ventral]
    # rois_ventral_R = [f'R_{roi}' for roi in rois_ventral]
    # rois_ventral_LR = rois_ventral_L + rois_ventral_R

    # mask_ventral_LR = atlas.get_mask(rois_ventral_LR)[0]
    # calc_RSM1(
    #     mask=mask_ventral_LR,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-ventral-LR.pkl')
    # )
    # calc_RSM1(
    #     mask=np.logical_and(R2_mask, mask_ventral_LR),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-ventral-LR_R2.pkl')
    # )
    # HCP-MMP-visual2 4+13+14 groups (ventral)<<<

    # >>>HCP-MMP-visual2's No.5 group (middle)
    # rois_middle = get_rois('MMP-vis2-G5')
    # print('rois_middle:', rois_middle)
    # rois_middle_L = [f'L_{roi}' for roi in rois_middle]
    # rois_middle_R = [f'R_{roi}' for roi in rois_middle]
    # rois_middle_LR = rois_middle_L + rois_middle_R

    # mask_middle_LR = atlas.get_mask(rois_middle_LR)[0]
    # calc_RSM1(
    #     mask=mask_middle_LR,
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-middle-LR.pkl')
    # )
    # calc_RSM1(
    #     mask=np.logical_and(R2_mask, mask_middle_LR),
    #     out_file=pjoin(work_dir, 'RSM_MMP-vis2-middle-LR_R2.pkl')
    # )
    # HCP-MMP-visual2's No.5 group (middle)<<<

    # >>>MMP-vis3-R PC1层级mask
    N = 2
    R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    pc1_mask = nib.load(pjoin(
        anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_N{N}.dlabel.nii'
    )).get_fdata()[0]
    src_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )

    for n in range(1, N+1):
        mask = pc1_mask == n
        calc_RSM1(
            src_file=src_file, mask=mask,
            out_file=pjoin(work_dir, f'RSM_MMP-vis3-R_PC1-N{N}-{n}.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask),
            out_file=pjoin(work_dir, f'RSM_MMP-vis3-R_PC1-N{N}-{n}_R2.pkl')
        )
    # MMP-vis3-R PC1层级mask<<<

    # calc_RSM2()
