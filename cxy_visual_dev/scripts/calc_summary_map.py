import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from scipy.stats.stats import sem
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'summary_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def make_mean_map(src_file, out_file):
    reader = CiftiReader(src_file)
    data = np.nanmean(reader.get_data(), 0, keepdims=True)
    save2cifti(out_file, data, reader.brain_models(), volume=reader.volume)


def make_age_maps(src_file, info_file, age_name, metric, out_file,
                  nan_mode=False):
    """
    对每个点，summary各年龄段被试的map

    Args:
        src_file (str): end with .dscalar.nii
        info_file (str): subject info file
        age_name (str): column name of age in info_file
        metric (str): summary method
        out_file (str): filename to save
        nan_mode (bool):
            If False, 没点要么全都是nan，要么没有nan
            If True，说明存在某个被试的所有点都是nan的情况
                在去掉这些被试之后仍然满足False时的条件
    """
    # prepare
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()
    n_vtx = src_maps.shape[1]

    info_df = pd.read_csv(info_file)
    ages = np.array(info_df[age_name])
    ages_uniq = np.unique(ages)
    n_age = len(ages_uniq)

    if metric == 'mean':
        func = np.mean
    elif metric == 'sem':
        func = sem
    else:
        raise ValueError('not supported metric:', metric)

    # calculate
    out_maps = np.ones((n_age, n_vtx)) * np.nan
    map_names = []

    if nan_mode:
        nan_idx_arr = np.isnan(src_maps)
        all_nan_vec = np.all(nan_idx_arr, 1)
        src_maps = src_maps[~all_nan_vec]
        ages = ages[~all_nan_vec]

    for age_idx, age in enumerate(ages_uniq):
        data = src_maps[ages == age]
        out_maps[age_idx] = func(data, 0)
        map_names.append(str(age))

    # save
    save2cifti(out_file, out_maps, reader.brain_models(), map_names, reader.volume)


if __name__ == '__main__':
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-alff.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-alff_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-falff.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-falff_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_MMP-vis3.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-GBC_MMP-vis3_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC1.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-GBC1_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength1.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength1_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_cortex.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-GBC_cortex_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength_cortex.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength_cortex_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_subcortex.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-GBC_subcortex_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength_subcortex.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength_subcortex_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength_R.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength_R_mean.dscalar.nii')
    # )
    make_mean_map(
        src_file=pjoin(proj_dir, 'data/HCP/HCPY_corrThickness_mine.dscalar.nii'),
        out_file=pjoin(work_dir, 'HCPY_corrThickness_mean_mine.dscalar.nii')
    )

    # make_age_maps(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
    #     age_name='age in years', metric='mean',
    #     out_file=pjoin(work_dir, 'HCPD-thickness_age-map-mean.dscalar.nii')
    # )
    # make_age_maps(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
    #     age_name='age in years', metric='sem',
    #     out_file=pjoin(work_dir, 'HCPD-thickness_age-map-sem.dscalar.nii')
    # )
    # make_age_maps(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #     info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
    #     age_name='age in years', metric='mean',
    #     out_file=pjoin(work_dir, 'HCPD-myelin_age-map-mean.dscalar.nii')
    # )
