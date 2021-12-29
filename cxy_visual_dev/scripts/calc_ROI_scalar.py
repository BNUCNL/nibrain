import os
import pandas as pd
from os.path import join as pjoin
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import proj_dir, Atlas
from magicbox.io.io import CiftiReader

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'ROI_scalar')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def ROI_scalar(src_file, mask, values, metric, out_file, rois=None, out_index=None):
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

    out_data = summary_across_col_by_mask(src_maps, mask, values, metric)
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


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')
    ROI_scalar(
        src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
        mask=atlas.maps[0], values=list(atlas.roi2label.values()),
        metric='mean', rois=list(atlas.roi2label.keys()),
        out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP.csv')
    )
