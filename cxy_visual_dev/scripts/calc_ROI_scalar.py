import os
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import get_rois, proj_dir, Atlas
from magicbox.io.io import CiftiReader

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'ROI_scalar')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def ROI_scalar(src_file, atlas, metric, out_file, rois='all', out_index=None):
    """
    为每个被试每个ROI求scalar value

    Args:
        src_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        atlas (Atlas): include ROIs' labels and mask map
        metric (str): mean, sum, sem
        out_file (str): end with .csv
        rois (str | strings): ROI names
            If is str, must be 'all'.
        out_index (None | str | sequence):
            If None, don't save index to out_file.
            If str, must be 'map_name' that means using map names as indices.
            If sequence, its length is equal to n_map.
    """
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()

    if rois == 'all':
        rois = list(atlas.roi2label.keys())
        values = list(atlas.roi2label.values())
    else:
        values = [atlas.roi2label[i] for i in rois]

    out_data = summary_across_col_by_mask(
        src_maps, atlas.maps[0], values, metric)
    out_df = pd.DataFrame(out_data, columns=rois)

    if out_index is None:
        out_df.to_csv(out_file, index=False)
    elif out_index == 'map_name':
        out_df.index = reader.map_names()
        out_df.to_csv(out_file, index=True)
    else:
        assert len(out_index) == out_df.shape[0]
        out_df.index = out_index
        out_df.to_csv(out_file, index=True)


if __name__ == '__main__':
    ROI_scalar(
        src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP_new.csv')
    )
    # ROI_scalar(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     atlas=Atlas('HCP-MMP'), metric='mean',
    #     out_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP.csv')
    # )
    # ROI_scalar(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPA_myelin.dscalar.nii'),
    #     atlas=Atlas('HCP-MMP'), metric='mean',
    #     out_file=pjoin(work_dir, 'HCPA-myelin_HCP-MMP.csv')
    # )
    # ROI_scalar(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii'),
    #     atlas=Atlas('HCP-MMP'), metric='mean',
    #     out_file=pjoin(work_dir, 'HCPA-thickness_HCP-MMP.csv')
    # )
    ROI_scalar(
        src_file=pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'gdist_src-OccipitalPole_HCP-MMP.csv')
    )
    ROI_scalar(
        src_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_HCP-MMP.csv'),
        rois=get_rois('MMP-vis3-R'), out_index='map_name'
    )
