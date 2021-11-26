import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, variation
from cxy_visual_dev.lib.predefine import proj_dir, Atlas, get_rois,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k,\
    mmp_map_file, s1200_avg_R2
from magicbox.io.io import save2cifti, CiftiReader
from magicbox.algorithm.plot import plot_bar

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'tmp')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def C2_corr_ecc_angle_area():
    """
    在指定atlas的各个area内做C2和eccentricity以及polar angle的相关
    """

    # MMP-vis2-area
    atlas = Atlas('MMP-vis2-area')
    out_file = pjoin(work_dir, 'PCA-C2_corr_ecc+angle_MMP-vis2-area.dscalar.nii')

    # wang-vis-area
    # atlas = Atlas('wang-vis-area')
    # out_file = pjoin(work_dir, 'PCA-C2_corr_ecc+angle_wang-vis-area.dscalar.nii')

    src_file = pjoin(
        anal_dir, 'decomposition/HCPY-M+T_L+R_MMP_vis2_zscore1-split_PCA-subj.dscalar.nii')

    src_map = nib.load(src_file).get_fdata()[1]
    ecc_map = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k]
    angle_map = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k]
    reader = CiftiReader(mmp_map_file)

    data = np.ones((4, LR_count_32k), np.float64) * np.nan
    map_names = ('eccentricity', 'polar angle', 'ecc > angle', 'ecc > angle (p)')
    for lbl in atlas.roi2label.values():
        roi_idx_map = atlas.maps[0] == lbl
        src_vec = src_map[roi_idx_map]
        ecc_vec = ecc_map[roi_idx_map]
        angle_vec = angle_map[roi_idx_map]

        nan_vec = np.zeros_like(src_vec, bool)
        for vec in (src_vec, ecc_vec, angle_vec):
            nan_vec = np.logical_or(nan_vec, np.isnan(vec))
        if np.all(nan_vec):
            continue
        non_nan_vec = ~nan_vec

        src_vec = src_vec[non_nan_vec]
        ecc_vec = ecc_vec[non_nan_vec]
        angle_vec = angle_vec[non_nan_vec]

        ecc_r, ecc_p = pearsonr(src_vec, ecc_vec)
        data[0, roi_idx_map] = ecc_r
        angle_r, angle_p = pearsonr(src_vec, angle_vec)
        data[1, roi_idx_map] = angle_r

        if np.abs(ecc_r) > np.abs(angle_r):
            data[2, roi_idx_map] = 1
            if ecc_p < 0.05 or angle_p < 0.05:
                data[3, roi_idx_map] = 1
        elif np.abs(ecc_r) == np.abs(angle_r):
            data[2, roi_idx_map] = 0
            if ecc_p < 0.05 or angle_p < 0.05:
                data[3, roi_idx_map] = 0
        else:
            data[2, roi_idx_map] = -1
            if ecc_p < 0.05 or angle_p < 0.05:
                data[3, roi_idx_map] = -1

    save2cifti(out_file, data, reader.brain_models(), map_names)


def make_EDMV_dlabel():
    """
    依据HCP MMP的22组，将HCP-MMP-visual2分成四份：
    Early: Group1+2
    Dorsal: Group3+16+17+18
    Middle: Group5
    Ventral: Group4+13+14
    将其制作成.dlabel.nii文件
    """
    reader = CiftiReader(mmp_map_file)
    atlas = Atlas('HCP-MMP')
    out_file = pjoin(work_dir, 'MMP-vis2-EDMV.dlabel.nii')

    data = np.ones((1, LR_count_32k), np.float64) * np.nan
    lbl_tab = nib.cifti2.Cifti2LabelTable()

    rois_early = get_rois('MMP-vis2-G1') + get_rois('MMP-vis2-G2')
    rois_early_L = [f'L_{roi}' for roi in rois_early]
    rois_early_R = [f'R_{roi}' for roi in rois_early]
    rois_early_LR = rois_early_L + rois_early_R
    mask_early_LR = atlas.get_mask(rois_early_LR)
    data[mask_early_LR] = 1
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'early', 1, 0, 0, 1)

    rois_dorsal = get_rois('MMP-vis2-G3') + get_rois('MMP-vis2-G16') +\
        get_rois('MMP-vis2-G17') + get_rois('MMP-vis2-G18')
    rois_dorsal_L = [f'L_{roi}' for roi in rois_dorsal]
    rois_dorsal_R = [f'R_{roi}' for roi in rois_dorsal]
    rois_dorsal_LR = rois_dorsal_L + rois_dorsal_R
    mask_dorsal_LR = atlas.get_mask(rois_dorsal_LR)
    data[mask_dorsal_LR] = 2
    lbl_tab[2] = nib.cifti2.Cifti2Label(2, 'dorsal', 0, 1, 0, 1)

    rois_middle = get_rois('MMP-vis2-G5')
    rois_middle_L = [f'L_{roi}' for roi in rois_middle]
    rois_middle_R = [f'R_{roi}' for roi in rois_middle]
    rois_middle_LR = rois_middle_L + rois_middle_R
    mask_middle_LR = atlas.get_mask(rois_middle_LR)
    data[mask_middle_LR] = 3
    lbl_tab[3] = nib.cifti2.Cifti2Label(3, 'middle', 0, 0, 1, 1)

    rois_ventral = get_rois('MMP-vis2-G4') + get_rois('MMP-vis2-G13') +\
        get_rois('MMP-vis2-G14')
    rois_ventral_L = [f'L_{roi}' for roi in rois_ventral]
    rois_ventral_R = [f'R_{roi}' for roi in rois_ventral]
    rois_ventral_LR = rois_ventral_L + rois_ventral_R
    mask_ventral_LR = atlas.get_mask(rois_ventral_LR)
    data[mask_ventral_LR] = 4
    lbl_tab[4] = nib.cifti2.Cifti2Label(4, 'ventral', 1, 1, 0, 1)

    save2cifti(out_file, data, reader.brain_models(), label_tables=[lbl_tab])


def make_R2_thr98_mask():
    """
    将S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii
    在9.8以上的阈上部分做成mask，存为dlabel文件
    """
    reader = CiftiReader(mmp_map_file)
    r2_map = nib.load(s1200_avg_R2).get_fdata()[:, :LR_count_32k]
    data = np.zeros((1, LR_count_32k), np.uint8)
    data[r2_map > 9.8] = 1
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, 'Subthreshold', 1, 1, 1, 0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'Suprathreshold', 1, 0, 0, 1)
    save2cifti(pjoin(work_dir, 'R2-thr9.8.dlabel.nii'), data,
               reader.brain_models(), label_tables=[lbl_tab])


def make_va_MMP_vis2():
    """
    用MMP-vis2 mask卡一下vertex area
    """
    mask = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))
    reader = CiftiReader(mmp_map_file)
    _, _, idx2v_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT')
    _, _, idx2v_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT')

    va_l = nib.load('/nfs/p1/public_dataset/datasets/hcp/DATA/'
                    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                    'S1200.L.midthickness_MSMAll_va.32k_fs_LR.shape.gii').darrays[0].data
    va_r = nib.load('/nfs/p1/public_dataset/datasets/hcp/DATA/'
                    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                    'S1200.R.midthickness_MSMAll_va.32k_fs_LR.shape.gii').darrays[0].data
    data = np.r_[va_l[idx2v_l], va_r[idx2v_r]][None, :]
    data[~mask] = np.nan
    save2cifti(pjoin(work_dir, 'va_MMP-vis2.dscalar.nii'), data, reader.brain_models())


def calc_variation():
    """
    对于PC1，我们已经认定它是从后到前渐变的梯度，并且用以枕极为锚点的距离作为PC1的理想模型。
    依据该距离分段，在同一距离段内的顶点属于一个层级。可以看到PC1的主要变异是层级间的渐变。
    反映的是从低级视觉到高级视觉这个整体的功能分化。

    PC2作为去除PC1之后的主要成分，开始呈现出层级内的变异，但看起来也存在层级间的变异,
    只是不是渐变，而是和层级内变异一样的波动着的变异。或许和局部功能分化有关。

    分析思路：
    将离枕极的距离从最小值到最大值分为N个层级（N等分），计算第i层的变异（var_i）和均值(mean_i)
    计算N个均值的变异作为层间变异（var_between），计算N个变异的均值作为层内变异（var_within）

    预期结果：
    PC2的层内变异要远大于PC1
    PC1的层间变异要远大于层内
    PC2层内和层间变异都较大，最好是层内大于层间
    """
    # prepare parameters
    n_segment = 10
    method = 'CV3'  # CV1, CV2, CV3, std
    n_pc = 2  # 前N个成分
    pc_names = ('C1', 'C2')
    title = f'segment{n_segment}_{method}'
    out_file1 = pjoin(work_dir, f'{title}_1.jpg')
    out_file2 = pjoin(work_dir, f'{title}_2.jpg')

    # prepare mask
    mask = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0]

    # prepare geodesic distance and segment boundaries
    gdist_file = pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii')
    # gdist_file = pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii')  # 效果还是以枕极为锚点比较好
    gdist_map = nib.load(gdist_file).get_fdata()[0, mask]
    min_gdist, max_gdist = np.min(gdist_map), np.max(gdist_map)
    segment_boundaries = np.linspace(min_gdist, max_gdist, n_segment + 1)

    # prepare PC maps
    pc_file = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc, mask]
    if method == 'CV1':
        var_func = variation
    elif method == 'CV2':
        # 每个PC都减去各自的最小值
        pc_maps = pc_maps - np.min(pc_maps, 1, keepdims=True)
        var_func = variation
    elif method == 'CV3':
        # 用绝对值计算作为分母的均值（标准差还是基于原数据计算）
        def var_func(arr, axis=None, ddof=0):
            var = np.std(arr, axis, ddof=ddof) /\
                np.mean(np.abs(arr), axis)
            return var
    elif method == 'std':
        var_func = np.std
    else:
        raise ValueError

    # calculating
    segment_means = np.zeros((n_pc, n_segment), np.float64)
    segment_vars = np.zeros((n_pc, n_segment), np.float64)
    for s_idx, s_boundary in enumerate(segment_boundaries[:-1]):
        e_idx = s_idx + 1
        e_boundary = segment_boundaries[e_idx]
        if e_idx == n_segment:
            segment_mask = np.logical_and(
                gdist_map >= s_boundary, gdist_map <= e_boundary)
        else:
            segment_mask = np.logical_and(
                gdist_map >= s_boundary, gdist_map < e_boundary)
        segments = pc_maps[:, segment_mask]
        segment_means[:, s_idx] = np.mean(segments, 1)
        segment_vars[:, s_idx] = var_func(segments, 1)
    var_along = var_func(segment_means, 1)
    var_vertical = np.mean(segment_vars, 1)

    plot_bar(np.array([var_along, var_vertical]), figsize=(4, 4),
             label=('var_along', 'var_vertical'), xticklabel=pc_names,
             ylabel='variation', title=title, mode=out_file1)
    plot_bar(segment_vars, figsize=(8, 4), label=pc_names,
             xticklabel=np.arange(1, n_segment+1),
             ylabel='variation', title=title, mode=out_file2)


if __name__ == '__main__':
    # C2_corr_ecc_angle_area()
    # make_EDMV_dlabel()
    # make_R2_thr98_mask()
    # make_va_MMP_vis2()
    calc_variation()
