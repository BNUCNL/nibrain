import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import proj_dir, Atlas, s1200_avg_RFsize,\
    s1200_avg_eccentricity, s1200_avg_angle, LR_count_32k, get_rois
from magicbox.io.io import CiftiReader

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'ROI_scalar')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def ROI_scalar(src_file, mask, values, metric, out_file, zscore_flag=False,
               rois=None, out_index=None):
    """
    为每个被试每个ROI求scalar value

    Args:
        src_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        mask (1D array): mask map
        values (sequence):
        metric (str): mean, sum, sem
        out_file (str): end with .csv
        rois (strings): ROI names corresponding to values
        out_index (None | str | sequence):
            If None, don't save index to out_file.
            If str, must be 'map_name' that means using map names as indices.
            If sequence, its length is equal to n_map.
    """
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()
    columns = values if rois is None else rois

    mask_all = np.zeros_like(mask, bool)
    for i in values:
        mask_all = np.logical_or(mask_all, mask == i)
    src_maps = src_maps[:, mask_all]
    mask = mask[mask_all]

    out_data = summary_across_col_by_mask(src_maps, mask, values, metric,
                                          zscore_flag=zscore_flag)
    out_df = pd.DataFrame(out_data, columns=columns)

    if out_index is None:
        out_df.to_csv(out_file, index=False)
    elif out_index == 'map_name':
        out_df.index = reader.map_names()
        out_df.to_csv(out_file, index=True)
    else:
        assert len(out_index) == out_df.shape[0]
        out_df.index = out_index
        out_df.to_csv(out_file, index=True)


def ROI_scalar1(atlas_name, Hemi):
    """
    为各map计算视觉ROI内的各种metric
    """
    roi_type = f'{atlas_name}-{Hemi}'
    if atlas_name == 'MMP-vis3':
        atlas = Atlas('HCP-MMP')
        rois = get_rois(roi_type)
    elif atlas_name == 'Wang2015':
        atlas = Atlas('Wang2015')
        rois = get_rois(roi_type)
    else:
        raise ValueError('unsupported atlas_name:', atlas_name)

    values = [atlas.roi2label[i] for i in rois]
    metrics = ['mean', 'std', 'sem', 'cv', 'sum']
    out_file = pjoin(work_dir, f'ROI_scalar1_{roi_type}.csv')

    # 结构梯度的PC1, PC2: stru-C1, stru-C2;
    map_stru_pc = nib.load(pjoin(
        anal_dir, f'decomposition/HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]
    map_names = ['PC1', 'PC2']
    maps = [map_stru_pc]

    # 离距状沟的距离: distFromCS;
    map_dist_cs = nib.load(pjoin(
        anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'
    )).get_fdata()
    map_names.extend(['distFromCS'])
    maps.extend([map_dist_cs])

    # Eccentricity; PolarAngle; RFsize;
    map_ecc = nib.load(s1200_avg_eccentricity).get_fdata()[0, :LR_count_32k][None, :]
    map_ang = nib.load(s1200_avg_angle).get_fdata()[0, :LR_count_32k][None, :]
    map_rfs = nib.load(s1200_avg_RFsize).get_fdata()[0, :LR_count_32k][None, :]
    map_names.extend(['Eccentricity', 'Angle', 'RFsize'])
    maps.extend([map_ecc, map_ang, map_rfs])

    # calculate
    maps = np.concatenate(maps, 0)
    out_data = summary_across_col_by_mask(
        data=maps, mask=atlas.maps[0], values=values, metrics=metrics,
        tol_size=10, nan_mode=True, row_names=map_names, zscore_flag=False, out_dict=False)

    # save out
    out_data = np.concatenate(out_data, 0)
    out_indices = []
    for metric in metrics:
        out_indices.extend([f'{metric}_{i}' for i in map_names])
    out_df = pd.DataFrame(out_data, out_indices, rois)
    out_df.to_csv(out_file, index=True)


if __name__ == '__main__':
    # atlas = Atlas('HCP-MMP')
    # ROI_scalar(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #     mask=atlas.maps[0], values=list(atlas.roi2label.values()),
    #     metric='mean', rois=list(atlas.roi2label.keys()),
    #     out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP.csv')
    # )

    # Ns = (3,)
    # for N in Ns:
    #     reader = CiftiReader(pjoin(
    #         anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_N{N}.dlabel.nii'
    #     ))
    #     mask_maps = reader.get_data()
    #     for i, name in enumerate(reader.map_names()):
    #         ROI_scalar(
    #             src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #             mask=mask_maps[i], values=np.arange(1, N+1), metric='mean',
    #             out_file=pjoin(work_dir, f'HCPD-myelin_N{N}-{name}.csv')
    #         )

    # N = 3
    # mask_map = nib.load(pjoin(
    #     anal_dir, 'mask_map/'
    #     f'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_{N}x{N}.dlabel.nii'
    # )).get_fdata()[0]
    # ROI_scalar(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #     mask=mask_map, values=np.arange(1, N*N+1), metric='mean',
    #     out_file=pjoin(work_dir, f'HCPD-myelin_{N}x{N}.csv')
    # )

    ROI_scalar1(atlas_name='MMP-vis3', Hemi='R')
    ROI_scalar1(atlas_name='Wang2015', Hemi='R')
    ROI_scalar1(atlas_name='MMP-vis3', Hemi='L')
    ROI_scalar1(atlas_name='Wang2015', Hemi='L')
