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
                if np.sum(non_nan_vec) < 2:
                    r, p = np.nan, np.nan
                else:
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
    计算各map之间的对称相关矩阵, (基调是: MMP-vis3-R)
    """
    # 结构梯度的PC1, PC2: stru-C1, stru-C2;
    map_stru_pc = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:2, mask]
    map_names = ['stru-C1', 'stru-C2']
    maps = [map_stru_pc]

    # 离距状沟(旧), (新)的距离: distFromCS1, distFromCS2;
    # 离枕极, MT的距离: distFromOP, distFromMT;
    map_dist_cs1 = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_cs2 = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus-split.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_op = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_dist_mt = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-MT.dscalar.nii'
    )).get_fdata()[0, mask][None, :]
    map_names.extend(['distFromCS1', 'distFromCS2', 'distFromOP', 'distFromMT'])
    maps.extend([map_dist_cs1, map_dist_cs2, map_dist_op, map_dist_mt])

    # Curvature; VertexArea;
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
    map_names.extend(['Curvature', 'VertexArea'])
    maps.extend([map_curv, map_va])

    # Eccentricity; PolarAngle; RFsize;
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_ang = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_rfs = nib.load(s1200_avg_RFsize).get_fdata()[0, :LR_count_32k][mask][None, :]
    map_names.extend(['Eccentricity', 'Angle', 'RFsize'])
    maps.extend([map_ecc, map_ang, map_rfs])

    # 周明的PC1~4;
    map_zm = nib.load(pjoin(
        proj_dir, 'data/space/zm_PCs.dscalar.nii')).get_fdata()[:, mask]
    map_names.extend(['ZM-PC1', 'ZM-PC2', 'ZM-PC3', 'ZM-PC4'])
    maps.append(map_zm)

    # RSFC_MMP-vis3-R2cortex_PCA-comp和RSFC_MMP-vis3-R2cortex_PCA-weight的PC1~6;
    map_rsfc_comp = nib.load(pjoin(
        anal_dir, 'decomposition/RSFC_MMP-vis3-R2cortex_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, mask]
    map_names.extend(f'RSFC-C{i}' for i in range(1, 7))
    maps.append(map_rsfc_comp)

    map_rsfc_weight = nib.load(pjoin(
        anal_dir, 'decomposition/RSFC_MMP-vis3-R2cortex_PCA-weight.dscalar.nii'
    )).get_fdata()[:6, mask]
    map_names.extend(f'RSFC-W{i}' for i in range(1, 7))
    maps.append(map_rsfc_weight)

    # RSFC_MMP-vis3-R2cortex_zscore_PCA-comp和RSFC_MMP-vis3-R2cortex_zscore_PCA-weight的PC1~6
    map_rsfc_zscore_comp = nib.load(pjoin(
        anal_dir, 'decomposition/RSFC_MMP-vis3-R2cortex_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, mask]
    map_names.extend(f'RSFC-zscore-C{i}' for i in range(1, 7))
    maps.append(map_rsfc_zscore_comp)

    map_rsfc_zscore_weight = nib.load(pjoin(
        anal_dir, 'decomposition/RSFC_MMP-vis3-R2cortex_zscore_PCA-weight.dscalar.nii'
    )).get_fdata()[:6, mask]
    map_names.extend(f'RSFC-zscore-W{i}' for i in range(1, 7))
    maps.append(map_rsfc_zscore_weight)

    # S1200-grp-RSFC-z_grayordinate2grayordinate_PCA-comp和
    # S1200-grp-RSFC-z_MMP-vis3-R2grayordinate_PCA-comp的PC1~6;
    map_grp_rsfc_all_comp = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-z_grayordinate2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-z-all-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_all_comp)

    map_grp_rsfc_vis_comp = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-z_MMP-vis3-R2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-z-vis-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_comp)

    # S1200-grp-RSFC-r_grayordinate2grayordinate_PCA-comp和
    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp的PC1~6;
    map_grp_rsfc_all_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_grayordinate2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-all-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_all_comp1)

    map_grp_rsfc_vis_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-vis-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_comp1)

    # S1200-grp-RSFC-r_cortex2cortex_PCA-comp和
    # S1200-grp-RSFC-r_MMP-vis3-R2cortex_PCA-comp的PC1~6;
    map_grp_rsfc_c2c_comp = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_cortex2cortex_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-c2c-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_c2c_comp)

    map_grp_rsfc_v2c_comp = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2cortex_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-v2c-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_v2c_comp)

    # 各频段震荡幅度(aff), (faff);
    reader = CiftiReader(pjoin(anal_dir, 'AFF/HCPY-aff.dscalar.nii'))
    map_aff = reader.get_data()[:, :LR_count_32k][:, mask]
    map_names.extend(f'A{i}' for i in reader.map_names())
    maps.append(map_aff)

    reader = CiftiReader(pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii'))
    map_faff = reader.get_data()[:, :LR_count_32k][:, mask]
    map_names.extend(f'fA{i}' for i in reader.map_names())
    maps.append(map_faff)

    # S1200-grp-RSFC-r_grayordinate2grayordinate_zscore_PCA-comp和
    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp的PC1~6
    map_grp_rsfc_all_z_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_grayordinate2grayordinate_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-all-z-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_all_z_comp1)

    map_grp_rsfc_vis_z_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'grp-RSFC-r-vis-z-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_z_comp1)

    # HCPY-avg_RSFC-MMP-vis3-R2grayordinate_PCA-comp和
    # HCPY-avg_RSFC-MMP-vis3-R2grayordinate_PCA-weight的PC1~6;
    map_rsfc_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-avg_RSFC-MMP-vis3-R2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'avg-RSFC-vis-C{i}' for i in range(1, 7))
    maps.append(map_rsfc_comp1)

    map_rsfc_weight1 = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-avg_RSFC-MMP-vis3-R2grayordinate_PCA-weight.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'avg-RSFC-vis-W{i}' for i in range(1, 7))
    maps.append(map_rsfc_weight1)

    # HCPY-avg_RSFC-MMP-vis3-R2grayordinate_zscore_PCA-comp和
    # HCPY-avg_RSFC-MMP-vis3-R2grayordinate_zscore_PCA-weight的PC1~6;
    map_rsfc_comp2 = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-avg_RSFC-MMP-vis3-R2grayordinate_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'avg-RSFC-vis-z-C{i}' for i in range(1, 7))
    maps.append(map_rsfc_comp2)

    map_rsfc_weight2 = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-avg_RSFC-MMP-vis3-R2grayordinate_zscore_PCA-weight.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k][:, mask]
    map_names.extend(f'avg-RSFC-vis-z-W{i}' for i in range(1, 7))
    maps.append(map_rsfc_weight2)

    # calculation
    maps = np.concatenate(maps, 0)
    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps, True)
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM1_main(mask_name):

    if mask_name == 'MMP-vis3-R':
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
        mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]

        calc_RSM1(
            mask=mask,
            out_file=pjoin(work_dir, f'RSM1_{mask_name}.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask),
            out_file=pjoin(work_dir, f'RSM1_{mask_name}_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-early+later':
        # 早期及其它视觉mask
        atlas = Atlas('HCP-MMP')
        rois_vis = get_rois('MMP-vis3-R')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        rois_early = get_rois('MMP-vis3-G1') + get_rois('MMP-vis3-G2')
        rois_early = [f'R_{roi}' for roi in rois_early]
        print('MMP-vis3-R-early:', rois_early)

        mask_early = atlas.get_mask(rois_early)[0]
        calc_RSM1(
            mask=mask_early,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_early),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early_R2.pkl')
        )

        rois_later = rois_vis.copy()
        for roi in rois_early:
            rois_later.remove(roi)
        mask_later = atlas.get_mask(rois_later)[0]
        calc_RSM1(
            mask=mask_later,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_later),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-early2+later2':
        # early2: V1~3
        # later2: 除V1~3以外的视觉区
        atlas = Atlas('HCP-MMP')
        rois_vis = get_rois('MMP-vis3-R')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        rois_early = ['R_V1', 'R_V2', 'R_V3']
        print('MMP-vis3-R-early2:', rois_early)

        mask_early = atlas.get_mask(rois_early)[0]
        calc_RSM1(
            mask=mask_early,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early2.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_early),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-early2_R2.pkl')
        )

        rois_later = rois_vis.copy()
        for roi in rois_early:
            rois_later.remove(roi)
        mask_later = atlas.get_mask(rois_later)[0]
        calc_RSM1(
            mask=mask_later,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later2.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_later),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-later2_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-V1/2/3/4':
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        for i in range(1, 5):
            mask = atlas.get_mask(f'R_V{i}')[0]
            calc_RSM1(
                mask=mask,
                out_file=pjoin(work_dir, f'RSM1_MMP-vis3-R-V{i}.pkl')
            )
            calc_RSM1(
                mask=np.logical_and(R2_mask, mask),
                out_file=pjoin(work_dir, f'RSM1_MMP-vis3-R-V{i}_R2.pkl')
            )

    elif mask_name == 'MMP-vis3-R-dorsal':
        # 3+16+17+18 groups
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        rois_dorsal = get_rois('MMP-vis3-G3') + get_rois('MMP-vis3-G16') +\
            get_rois('MMP-vis3-G17') + get_rois('MMP-vis3-G18')
        rois_dorsal = [f'R_{roi}' for roi in rois_dorsal]
        print('MMP-vis3-R-dorsal:', rois_dorsal)

        mask_dorsal = atlas.get_mask(rois_dorsal)[0]
        calc_RSM1(
            mask=mask_dorsal,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-dorsal.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_dorsal),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-dorsal_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-ventral':
        # 4+13+14 groups (ventral)
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        rois_ventral = get_rois('MMP-vis3-G4') + get_rois('MMP-vis3-G13') +\
            get_rois('MMP-vis3-G14')
        rois_ventral = [f'R_{roi}' for roi in rois_ventral]
        print('MMP-vis3-R-ventral:', rois_ventral)
    
        mask_ventral = atlas.get_mask(rois_ventral)[0]
        calc_RSM1(
            mask=mask_ventral,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-ventral.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_ventral),
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-ventral_R2.pkl')
        )

    elif mask_name == 'MMP-vis3-R-middle':
        # No.5 group (middle)
        atlas = Atlas('HCP-MMP')
        R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8

        rois_middle = get_rois('MMP-vis3-G5')
        rois_middle = [f'R_{roi}' for roi in rois_middle]
        print('MMP-vis3-R-middle:', rois_middle)

        mask_middle = atlas.get_mask(rois_middle)[0]
        calc_RSM1(
            mask=mask_middle,
            out_file=pjoin(work_dir, 'RSM1_MMP-vis3-R-middle.pkl')
        )
        calc_RSM1(
            mask=np.logical_and(R2_mask, mask_middle),
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


def calc_RSM5():
    """
    计算各map在每个视觉区内与eccentricity的相关
    """
    atlas = Atlas('HCP-MMP')
    rois_vis = get_rois('MMP-vis3-R')
    n_roi = len(rois_vis)
    out_file = pjoin(work_dir, 'RSM5_corr-ECC_area.pkl')

    # 结构梯度的PC1, PC2: stru-C1, stru-C2;
    map_stru_pc = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]
    map_names = ['stru-C1', 'stru-C2']
    maps = [map_stru_pc]

    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp的PC1~6;
    map_grp_rsfc_vis_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k]
    map_names.extend(f'grp-RSFC-r-vis-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_comp1)

    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp的PC1~6
    map_grp_rsfc_vis_z_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k]
    map_names.extend(f'grp-RSFC-r-vis-z-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_z_comp1)

    # eccentricity
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k]

    # calculation
    maps = np.concatenate(maps, 0)
    n_map = maps.shape[0]
    assert n_map == len(map_names)
    rs = np.zeros((n_map, n_roi))
    ps = np.zeros((n_map, n_roi))
    for roi_idx, roi in enumerate(rois_vis):
        mask = atlas.get_mask(roi)[0]
        roi_ecc = map_ecc[mask]
        for map_idx in range(n_map):
            roi_map = maps[map_idx, mask]
            r, p = pearsonr(roi_map, roi_ecc)
            rs[map_idx, roi_idx] = r
            ps[map_idx, roi_idx] = p

    data = {'row_name': map_names, 'col_name': rois_vis, 'r': rs, 'p': ps}
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM6():
    """
    计算各map和eccentricity在视觉区域间的相关
    all: 使用所有的视觉区域
    ex(V1~3): 除V1~3以外的视觉区域
    ex(V1~4): 除V1~4以外的视觉区域
    ex(V1~4+V3A): 除V1~4以及V3A以外的视觉区域
    """
    atlas = Atlas('HCP-MMP')
    rois_vis = get_rois('MMP-vis3-R')
    col_names = ['all', 'ex(V1~3)', 'ex(V1~4)', 'ex(V1~4+V3A)']
    col2exROIs = {
        'ex(V1~3)': ['R_V1', 'R_V2', 'R_V3'],
        'ex(V1~4)': ['R_V1', 'R_V2', 'R_V3', 'R_V4'],
        'ex(V1~4+V3A)': ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V3A']
    }
    n_col = len(col_names)
    out_file = pjoin(work_dir, 'RSM6_corr-ECC_area-between.pkl')

    # 结构梯度的PC1, PC2: stru-C1, stru-C2;
    map_stru_pc = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]
    map_names = ['stru-C1', 'stru-C2']
    maps = [map_stru_pc]

    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp的PC1~6;
    map_grp_rsfc_vis_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k]
    map_names.extend(f'grp-RSFC-r-vis-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_comp1)

    # S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp的PC1~6
    map_grp_rsfc_vis_z_comp1 = nib.load(pjoin(
        anal_dir, 'decomposition/S1200-grp-RSFC-r_MMP-vis3-R2grayordinate_zscore_PCA-comp.dscalar.nii'
    )).get_fdata()[:6, :LR_count_32k]
    map_names.extend(f'grp-RSFC-r-vis-z-C{i}' for i in range(1, 7))
    maps.append(map_grp_rsfc_vis_z_comp1)

    # Eccentricity
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k]

    # calculation
    maps = np.concatenate(maps, 0)
    n_map = maps.shape[0]
    assert n_map == len(map_names)
    rs = np.zeros((n_map, n_col))
    ps = np.zeros((n_map, n_col))
    for col_idx, col in enumerate(col_names):
        if col == 'all':
            rois = rois_vis
        else:
            rois = [i for i in rois_vis if i not in col2exROIs[col]]
        n_roi = len(rois)
        print(f'n_roi of {col}:', n_roi)
        for map_idx in range(n_map):
            map_vec = np.zeros(n_roi)
            ecc_vec = np.zeros(n_roi)
            for roi_idx, roi in enumerate(rois):
                mask = atlas.get_mask(roi)[0]
                map_vec[roi_idx] = np.mean(maps[map_idx, mask])
                ecc_vec[roi_idx] = np.mean(map_ecc[mask])
            r, p = pearsonr(map_vec, ecc_vec)
            rs[map_idx, col_idx] = r
            ps[map_idx, col_idx] = p

    data = {'row_name': map_names, 'col_name': col_names, 'r': rs, 'p': ps}
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM7(dataset_name, vis_name):
    """
    计算stru-PC1/2和各滑窗PC的相关
    """
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    hcpy_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    n_hcpy_pc = 2
    hcpda_file = pjoin(
        anal_dir, f'decomposition/{dataset_name}-M+T_{vis_name}_zscore1_PCA-subj_SW-width50-step10-merge.pkl')
    n_hcpda_pc = 10
    out_file = pjoin(work_dir, f'RSM7_M+T_{vis_name}_zscore1_PCA-subj_HCPY_corr_{dataset_name}_SW-width50-step10-merge.pkl')

    hcpda_data = pkl.load(open(hcpda_file, 'rb'))
    row_names = hcpda_data['component name']
    hcpy_pc_maps = nib.load(hcpy_file).get_fdata()[:2, mask]
    col_names = []
    rs = np.zeros((n_hcpda_pc, n_hcpy_pc * hcpda_data['n_win']))
    ps = np.zeros((n_hcpda_pc, n_hcpy_pc * hcpda_data['n_win']))
    for win_idx in range(hcpda_data['n_win']):
        win_id = win_idx + 1
        s_idx = win_idx * n_hcpy_pc
        e_idx = s_idx + n_hcpy_pc
        rs_tmp, ps_tmp = calc_pearson_r_p(hcpda_data[f'Win{win_id}_comp'], hcpy_pc_maps)
        rs[:, s_idx:e_idx] = rs_tmp
        ps[:, s_idx:e_idx] = ps_tmp
        col_names.extend([f'HCPY-C{i}_corr_Win{win_id}' for i in range(1, n_hcpy_pc + 1)])

    data = {'row_name': row_names, 'col_name': col_names, 'r': rs, 'p': ps, 
            'n_win': hcpda_data['n_win'], 'age in months': hcpda_data['age in months']}
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM8(dataset_name, local_name):
    """
    在各个局部计算stru-PC1/2和各滑窗PC的相关
    """
    if local_name == 'MMP-vis3-R-EDMV':
        vis_name = 'MMP-vis3-R'
        reader = CiftiReader(pjoin(anal_dir, 'tmp/MMP-vis3-EDMV.dlabel.nii'))
        vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
        local_mask = reader.get_data()[0, vis_mask]
        local_name2key = {}
        lbl_tab = reader.label_tables()[0]
        for k in lbl_tab.keys():
            if k == 0:
                continue
            local_name2key[lbl_tab[k].label] = k
    else:
        raise ValueError('not supported local name:', local_name)
    n_local = len(local_name2key)

    pc_names = ['C1', 'C2']
    n_pc = len(pc_names)
    hcpy_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    hcpda_file = pjoin(
        anal_dir, f'decomposition/{dataset_name}-M+T_{vis_name}_zscore1_PCA-subj_SW-width50-step10-merge.pkl')
    out_file = pjoin(work_dir, f'RSM8_M+T_{vis_name}_zscore1_PCA-subj_HCPY_corr_{dataset_name}_SW-width50-step10-merge.pkl')

    hcpda_data = pkl.load(open(hcpda_file, 'rb'))
    hcpy_pc_maps = nib.load(hcpy_file).get_fdata()[:n_pc, vis_mask]

    rs = np.zeros((n_pc * n_local, hcpda_data['n_win']))
    ps = np.zeros((n_pc * n_local, hcpda_data['n_win']))
    row_names = []
    col_names = [f'Win{i}' for i in range(1, hcpda_data['n_win'] + 1)]
    row_idx = 0
    for local_name, local_key in local_name2key.items():
        mask = local_mask == local_key
        for pc_idx, pc_name in enumerate(pc_names):
            row_names.append(f'{local_name} {pc_name}')
            hcpy_pc_map = hcpy_pc_maps[pc_idx][mask]
            for col_idx, col_name in enumerate(col_names):
                hcpda_pc_map = hcpda_data[f'{col_name}_comp'][pc_idx][mask]
                r, p = pearsonr(hcpy_pc_map, hcpda_pc_map)
                rs[row_idx, col_idx] = r
                ps[row_idx, col_idx] = p
            row_idx += 1

    data = {'row_name': row_names, 'col_name': col_names, 'r': rs, 'p': ps, 
            'n_win': hcpda_data['n_win'], 'age in months': hcpda_data['age in months']}
    pkl.dump(data, open(out_file, 'wb'))


def calc_RSM9():
    """
    用个体差异法计算各种指标之间的相关
    """
    n_subj = 1096
    out_file = pjoin(work_dir, 'RSM9.pkl')

    # ---PCA权重---
    pc_names = ('C1', 'C2')
    # HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj中myelin的权重及其绝对值
    weight_m_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_M.csv')
    weight_m_df = pd.read_csv(weight_m_file, usecols=pc_names)
    weight_m_arr = np.c_[weight_m_df, np.abs(weight_m_df)].T
    map_names = [f'{i}_weight_M' for i in pc_names]
    map_names.extend([f'{i}_abs(weight)_M' for i in pc_names])
    maps = [weight_m_arr]

    # HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj中thickness的权重及其绝对值
    weight_t_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_T.csv')
    weight_t_df = pd.read_csv(weight_t_file, usecols=pc_names)
    weight_t_arr = np.c_[weight_t_df, np.abs(weight_t_df)].T
    map_names.extend([f'{i}_weight_T' for i in pc_names])
    map_names.extend([f'{i}_abs(weight)_T' for i in pc_names])
    maps.append(weight_t_arr)

    # ---behavior measures---
    beh_file1 = '/nfs/m1/hcp/S1200_behavior.csv'
    beh_file2 = '/nfs/m1/hcp/S1200_behavior_restricted.csv'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    beh_df1 = pd.read_csv(beh_file1)
    beh_df2 = pd.read_csv(beh_file2)
    info_df = pd.read_csv(info_file)
    assert np.all(beh_df1['Subject'] == beh_df2['Subject'])
    # get all numeric data
    cols1 = [i for i in beh_df1.columns if is_numeric_dtype(beh_df1[i])]
    cols2 = [i for i in beh_df2.columns if is_numeric_dtype(beh_df2[i])]
    cols2.remove('Subject')
    beh_arr = np.c_[np.array(beh_df1[cols1], np.float64),
                    np.array(beh_df2[cols2], np.float64)]
    cols = cols1 + cols2
    # limited in 1096 subjects
    subj_ids_beh = beh_df1['Subject'].to_list()
    subj_indices = [subj_ids_beh.index(i) for i in info_df['subID']]
    beh_arr = beh_arr[subj_indices].T
    map_names.extend(cols)
    maps.append(beh_arr)

    # ---HCPY-M+T_fit_PC_subj-wise---
    mt_fit_pc_file = pjoin(anal_dir, 'fit/HCPY-M+T_fit_PC_subj-wise.pkl')
    mt_fit_pc_data = pkl.load(open(mt_fit_pc_file, 'rb'))
    mt_fit_pc_maps = np.zeros((len(mt_fit_pc_data), n_subj))
    for k_idx, k in enumerate(mt_fit_pc_data.keys()):
        mt_fit_pc_maps[k_idx] = mt_fit_pc_data[k]
        map_names.append(k)
    maps.append(mt_fit_pc_maps)

    # calculate correlation
    maps = np.concatenate(maps, 0)
    data = {'row_name': map_names, 'col_name': map_names}
    data['r'], data['p'] = calc_pearson_r_p(maps, maps, True)
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # calc_RSM1_main(mask_name='MMP-vis3-R')

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
    # calc_RSM5()
    # calc_RSM6()
    # calc_RSM7(dataset_name='HCPD', vis_name='MMP-vis3-R')
    # calc_RSM7(dataset_name='HCPA', vis_name='MMP-vis3-R')

    # calc_RSM8(dataset_name='HCPD', local_name='MMP-vis3-R-EDMV')
    # calc_RSM8(dataset_name='HCPA', local_name='MMP-vis3-R-EDMV')

    calc_RSM9()
