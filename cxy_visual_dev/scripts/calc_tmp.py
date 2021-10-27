import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    s1200_avg_angle, s1200_avg_eccentricity, LR_count_32k, mmp_map_file
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


if __name__ == '__main__':
    C2_corr_ecc_angle_area()
