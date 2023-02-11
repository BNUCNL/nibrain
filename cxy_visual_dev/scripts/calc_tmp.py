import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, permutation_test
from cxy_visual_dev.lib.predefine import proj_dir, Atlas, get_rois,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k,\
    mmp_map_file, s1200_avg_R2
from magicbox.io.io import save2cifti, CiftiReader

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


def calc_and_test_PC_corr_geo_model():
    Hemis = ('L', 'R')
    pc_names = ('PC1', 'PC2')
    pc_files = pjoin(
        anal_dir, 'decomposition/'
        'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    geo_file1 = pjoin(
        anal_dir, 'gdist/gdist_src-OP.dscalar.nii')
    geo_file2 = pjoin(
        anal_dir, 'gdist/gdist4_src-EDLV-seed-v1.dscalar.nii')

    def statistic(x, y):
        return pearsonr(x, y)[0]

    geo_map1 = nib.load(geo_file1).get_fdata()[0]
    geo_map2 = nib.load(geo_file2).get_fdata()[0]
    for Hemi in Hemis:
        mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
        pc_file = pc_files.format(Hemi=Hemi)
        pc_maps = nib.load(pc_file).get_fdata()[:2, mask]
        geo_maps = [geo_map1[mask], geo_map2[mask]]
        for pc_idx, pc_name in enumerate(pc_names):
            print(f'---{Hemi}H {pc_name} corr geometry model---')
            x = pc_maps[pc_idx]
            y = geo_maps[pc_idx]
            pmt_test = permutation_test(
                (x, y), statistic, permutation_type='pairings',
                vectorized=False, n_resamples=10000, alternative='two-sided',
                random_state=7)
            print('pmt_test.statistic:\n', pmt_test.statistic)
            print('pmt_test.pvalue:\n', pmt_test.pvalue)
            print("pearsonr(x, y, alternative='two-sided'):\n",
                  pearsonr(x, y, alternative='two-sided'))


if __name__ == '__main__':
    # C2_corr_ecc_angle_area()
    # make_EDMV_dlabel()
    # make_R2_thr98_mask()
    # make_va_MMP_vis2()
    calc_and_test_PC_corr_geo_model()
