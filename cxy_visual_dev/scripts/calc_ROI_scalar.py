import os
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import proj_dir, Atlas

work_dir = pjoin(proj_dir, 'analysis/ROI_scalar')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def ROI_scalar1(src_file, atlas, metric, out_file):
    src_maps = nib.load(src_file).get_fdata()
    out_data = summary_across_col_by_mask(
        src_maps, atlas.maps[0], atlas.roi2label.values(), metric)
    out_df = pd.DataFrame(out_data, columns=atlas.roi2label.keys())
    out_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    ROI_scalar1(
        src_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP.csv')
    )
    ROI_scalar1(
        src_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP.csv')
    )
    ROI_scalar1(
        src_file=pjoin(proj_dir, 'data/HCP/HCPA_myelin.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPA-myelin_HCP-MMP.csv')
    )
    ROI_scalar1(
        src_file=pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii'),
        atlas=Atlas('HCP-MMP'), metric='mean',
        out_file=pjoin(work_dir, 'HCPA-thickness_HCP-MMP.csv')
    )
