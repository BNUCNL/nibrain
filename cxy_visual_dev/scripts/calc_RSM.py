import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from pandas.api.types import is_numeric_dtype
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
    计算PCA的C1, C2; distFromCS; distFromCS-split; distFromOP; distFromMT;
    Curvature; VertexArea; Eccentricity; PolarAngle; RFsize;
    以及周明的PC1~4之间的相关矩阵。
    """
    map_PCA = nib.load(src_file).get_fdata()[:2, mask]

    map_dist_cs = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_cs1 = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus-split.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_op = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_mt = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-MT.dscalar.nii'
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

    maps = np.concatenate([
        map_PCA, map_dist_cs, map_dist_cs1, map_dist_op, map_dist_mt,
        map_curv, map_va, map_ecc, map_ang, map_rfs, map_zm], 0)
    map_names = (
        'PCA-C1', 'PCA-C2', 'distFromCS', 'distFromCS-split', 'distFromOP', 'distFromMT',
        'Curvature', 'VertexArea', 'Eccentricity', 'Angle', 'RFsize',
        'ZM-PC1', 'ZM-PC2', 'ZM-PC3', 'ZM-PC4')

    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps, True)
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM1_main(mask_name):

    if mask_name == 'MMP-vis3-R':
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        calc_RSM1(
            src_file=src_file, mask=mask,
            out_file=pjoin(work_dir, f'RSM1_{mask_name}.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask),
            out_file=pjoin(work_dir, f'RSM1_{mask_name}_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-early+later':
        # 早期及其它视觉mask
        atlas = Atlas('HCP-MMP')
        rois_vis = get_rois('MMP-vis3-R')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        rois_early = get_rois('MMP-vis3-G1') + get_rois('MMP-vis3-G2')
        rois_early = [f'R_{roi}' for roi in rois_early]
        print('MMP-vis3-R-early:', rois_early)

        mask_early = atlas.get_mask(rois_early)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_early,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_early),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early_R2.pkl')
        )

        rois_later = rois_vis.copy()
        for roi in rois_early:
            rois_later.remove(roi)
        mask_later = atlas.get_mask(rois_later)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_later,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_later),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-early2+later2':
        # early2: V1~3
        # later2: 除V1~3以外的视觉区
        atlas = Atlas('HCP-MMP')
        rois_vis = get_rois('MMP-vis3-R')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        rois_early = ['R_V1', 'R_V2', 'R_V3']
        print('MMP-vis3-R-early2:', rois_early)

        mask_early = atlas.get_mask(rois_early)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_early,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early2.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_early),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early2_R2.pkl')
        )

        rois_later = rois_vis.copy()
        for roi in rois_early:
            rois_later.remove(roi)
        mask_later = atlas.get_mask(rois_later)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_later,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later2.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_later),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later2_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-V1/2/3/4':
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        for i in range(1, 5):
            mask = atlas.get_mask(f'R_V{i}')[0]
            calc_RSM1(
                src_file=src_file, mask=mask,
                out_file=pjoin(work_dir, f'RSM1_MMP-vis3-R-V{i}.pkl')
            )
            calc_RSM1(
                src_file=src_file, mask=np.logical_and(R2_mask, mask),
                out_file=pjoin(work_dir, f'RSM1_MMP-vis3-R-V{i}_R2.pkl')
            )

    elif mask_name == 'MMP-vis3-R-dorsal':
        # 3+16+17+18 groups
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        rois_dorsal = get_rois('MMP-vis3-G3') + get_rois('MMP-vis3-G16') +\
            get_rois('MMP-vis3-G17') + get_rois('MMP-vis3-G18')
        rois_dorsal = [f'R_{roi}' for roi in rois_dorsal]
        print('MMP-vis3-R-dorsal:', rois_dorsal)

        mask_dorsal = atlas.get_mask(rois_dorsal)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_dorsal,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-dorsal.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_dorsal),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-dorsal_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-ventral':
        # 4+13+14 groups (ventral)
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        rois_ventral = get_rois('MMP-vis3-G4') + get_rois('MMP-vis3-G13') +\
            get_rois('MMP-vis3-G14')
        rois_ventral = [f'R_{roi}' for roi in rois_ventral]
        print('MMP-vis3-R-ventral:', rois_ventral)
    
        mask_ventral = atlas.get_mask(rois_ventral)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_ventral,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-ventral.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_ventral),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-ventral_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-middle':
        # No.5 group (middle)
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        src_file = pjoin(
            anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
        )

        rois_middle = get_rois('MMP-vis3-G5')
        rois_middle = [f'R_{roi}' for roi in rois_middle]
        print('MMP-vis3-R-middle:', rois_middle)

        mask_middle = atlas.get_mask(rois_middle)[0]
        calc_RSM1(
            src_file=src_file, mask=mask_middle,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-middle.pkl')
        )
        calc_RSM1(
            src_file=src_file, mask=np.logical_and(R2_mask, mask_middle),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-middle_R2.pkl')
        )

    else:
        raise ValueError(mask_name)


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


def calc_RSM3():
    """
    计算PC1和PC2的权重和HCPYA所有类型为数值的行为数据的相关
    """
    pc_weight_abs = True  # 在求相关之前，先把权重取绝对值，这个值越大可以，对梯度贡献越大（无论正负贡献）
    pc_names = ('C1', 'C2')
    weight_m_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_M.csv'
    )
    weight_t_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_T.csv'
    )
    beh_file1 = '/nfs/m1/hcp/S1200_behavior.csv'
    beh_file2 = '/nfs/m1/hcp/S1200_behavior_restricted.csv'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    if pc_weight_abs:
        out_file = pjoin(work_dir, 'HCPY_PC12-abs-corr-beh.pkl')
    else:
        out_file = pjoin(work_dir, 'HCPY_PC12-corr-beh.pkl')

    # get all numeric data
    beh_df1 = pd.read_csv(beh_file1)
    beh_df2 = pd.read_csv(beh_file2)
    assert np.all(beh_df1['Subject'] == beh_df2['Subject'])
    cols1 = [i for i in beh_df1.columns if is_numeric_dtype(beh_df1[i])]
    cols2 = [i for i in beh_df2.columns if is_numeric_dtype(beh_df2[i])]
    cols2.remove('Subject')
    beh_arr = np.c_[np.array(beh_df1[cols1], np.float64),
                    np.array(beh_df2[cols2], np.float64)]
    cols = cols1 + cols2

    # limited in 1096 subjects
    subj_ids_beh = beh_df1['Subject'].to_list()
    info_df = pd.read_csv(info_file)
    subj_indices = [subj_ids_beh.index(i) for i in info_df['subID']]
    beh_arr = beh_arr[subj_indices].T

    # get pc1 and pc2
    weight_m_df = pd.read_csv(weight_m_file, usecols=pc_names)
    weight_t_df = pd.read_csv(weight_t_file, usecols=pc_names)
    weight_arr = np.c_[np.array(weight_m_df), np.array(weight_t_df)].T
    if pc_weight_abs:
        weight_arr = np.abs(weight_arr)
    rows = [f'{i}_M' for i in pc_names] + [f'{i}_T' for i in pc_names]

    # calculate correlation
    data = {'row_name': rows, 'col_name': cols}
    data['r'], data['p'] = calc_pearson_r_p(weight_arr, beh_arr, True)
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM4(a_type='aff'):
    """
    计算PC1,PC2，和各频段震荡幅度map的相关
    """
    atlas = Atlas('HCP-MMP')
    mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]
    pc_names = ('C1', 'C2')
    pc_file = pjoin(
        anal_dir,
        'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )
    aff_file = pjoin(anal_dir, f'AFF/HCPY-{a_type}.dscalar.nii')
    out_file = pjoin(work_dir, f'HCPY_PC12-corr-{a_type}.pkl')

    map_PCA = nib.load(pc_file).get_fdata()[:2, mask]
    reader = CiftiReader(aff_file)
    map_aff = reader.get_data()[:, :LR_count_32k][:, mask]

    # calculate correlation
    data = {'row_name': pc_names, 'col_name': reader.map_names()}
    data['r'], data['p'] = calc_pearson_r_p(map_PCA, map_aff)
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM5():
    """
    计算PC1, PC2和eccentricity在每个视觉区域内的相关
    """
    n_pc = 2
    pc_names = [f'PC{i}' for i in range(1, n_pc + 1)]
    map_pcs = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:n_pc]
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k]
    out_file = pjoin(work_dir, 'RSM5_PC12-corr-ECC_area.pkl')

    atlas = Atlas('HCP-MMP')
    rois_vis = get_rois('MMP-vis3-R')
    n_roi = len(rois_vis)

    rs = np.zeros((n_pc, n_roi))
    ps = np.zeros((n_pc, n_roi))
    for pc_idx in range(n_pc):
        for roi_idx, roi in enumerate(rois_vis):
            mask = atlas.get_mask(roi)[0]
            roi_pc = map_pcs[pc_idx, mask]
            roi_ecc = map_ecc[mask]
            r, p = pearsonr(roi_pc, roi_ecc)
            rs[pc_idx, roi_idx] = r
            ps[pc_idx, roi_idx] = p

    data = {'row_name': pc_names, 'col_name': rois_vis, 'r': rs, 'p': ps}
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM6():
    """
    计算PC1, PC2和eccentricity在视觉区域间的相关
    all: 使用所有的视觉区域
    ex(V1~3): 除V1~3以外的视觉区域
    ex(V1~4): 除V1~4以外的视觉区域
    ex(V1~4+V3A): 除V1~4以及V3A以外的视觉区域
    """
    n_pc = 2
    pc_names = [f'PC{i}' for i in range(1, n_pc + 1)]
    map_pcs = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:n_pc]
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k]
    out_file = pjoin(work_dir, 'RSM6_PC12-corr-ECC_area-between.pkl')

    atlas = Atlas('HCP-MMP')
    rois_vis = get_rois('MMP-vis3-R')
    col_names = ['all', 'ex(V1~3)', 'ex(V1~4)', 'ex(V1~4+V3A)']
    col2exROIs = {
        'ex(V1~3)': ['R_V1', 'R_V2', 'R_V3'],
        'ex(V1~4)': ['R_V1', 'R_V2', 'R_V3', 'R_V4'],
        'ex(V1~4+V3A)': ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V3A']
    }
    n_col = len(col_names)

    rs = np.zeros((n_pc, n_col))
    ps = np.zeros((n_pc, n_col))
    for pc_idx in range(n_pc):
        for col_idx, col in enumerate(col_names):
            if col == 'all':
                rois = rois_vis
            else:
                rois = [i for i in rois_vis if i not in col2exROIs[col]]
            n_roi = len(rois)
            print('n_roi:', n_roi)
            pc_vec = np.zeros(n_roi)
            ecc_vec = np.zeros(n_roi)
            for roi_idx, roi in enumerate(rois):
                mask = atlas.get_mask(roi)[0]
                pc_vec[roi_idx] = np.mean(map_pcs[pc_idx, mask])
                ecc_vec[roi_idx] = np.mean(map_ecc[mask])
            r, p = pearsonr(pc_vec, ecc_vec)
            rs[pc_idx, col_idx] = r
            ps[pc_idx, col_idx] = p

    data = {'row_name': pc_names, 'col_name': col_names, 'r': rs, 'p': ps}
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    calc_RSM1_main(mask_name='MMP-vis3-R')
    # calc_RSM1_main(mask_name='MMP-vis3-R-early+later')
    # calc_RSM1_main(mask_name='MMP-vis3-R-early2+later2')
    # calc_RSM1_main(mask_name='MMP-vis3-R-V1/2/3/4')

    # >>>MMP-vis3-R PC1层级mask
    # N = 2
    # R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    # pc1_mask = nib.load(pjoin(
    #     anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_N{N}.dlabel.nii'
    # )).get_fdata()[0]
    # src_file = pjoin(
    #     anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    # )

    # for n in range(1, N+1):
    #     mask = pc1_mask == n
    #     calc_RSM1(
    #         src_file=src_file, mask=mask,
    #         out_file=pjoin(work_dir, f'RSM_MMP-vis3-R_PC1-N{N}-{n}.pkl')
    #     )
    #     calc_RSM1(
    #         src_file=src_file, mask=np.logical_and(R2_mask, mask),
    #         out_file=pjoin(work_dir, f'RSM_MMP-vis3-R_PC1-N{N}-{n}_R2.pkl')
    #     )
    # MMP-vis3-R PC1层级mask<<<

    # calc_RSM2()
    # calc_RSM3()
    # calc_RSM4(a_type='aff')
    # calc_RSM4(a_type='faff')
    # calc_RSM5()
    # calc_RSM6()
