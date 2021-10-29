import os
import numpy as np
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin
from cxy_visual_dev.lib.algo import mask_maps

work_dir = pjoin(proj_dir, 'analysis/mean_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def make_mean_map(src_file, out_file):
    reader = CiftiReader(src_file)
    data = np.nanmean(reader.get_data(), 0, keepdims=True)
    save2cifti(out_file, data, reader.brain_models())


if __name__ == '__main__':
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-alff.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-alff_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_MMP-vis2.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPY-GBC_MMP-vis2_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=s1200_1096_myelin,
    #     out_file=pjoin(work_dir, 'HCPY-myelin_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=s1200_1096_thickness,
    #     out_file=pjoin(work_dir, 'HCPY-thickness_mean.dscalar.nii')
    # )

    mask_maps(
        data_file=pjoin(work_dir, 'HCPY-alff_mean.dscalar.nii'),
        atlas_name='MMP-vis2-LR', roi_names=('L_MMP_vis2', 'R_MMP_vis2'),
        out_file=pjoin(work_dir, 'HCPY-alff_mean_mask-MMP-vis2-LR.dscalar.nii')
    )
    mask_maps(
        data_file=pjoin(work_dir, 'HCPY-myelin_mean.dscalar.nii'),
        atlas_name='MMP-vis2-LR', roi_names=('L_MMP_vis2', 'R_MMP_vis2'),
        out_file=pjoin(work_dir, 'HCPY-myelin_mean_mask-MMP-vis2-LR.dscalar.nii')
    )
    mask_maps(
        data_file=pjoin(work_dir, 'HCPY-thickness_mean.dscalar.nii'),
        atlas_name='MMP-vis2-LR', roi_names=('L_MMP_vis2', 'R_MMP_vis2'),
        out_file=pjoin(work_dir, 'HCPY-thickness_mean_mask-MMP-vis2-LR.dscalar.nii')
    )
