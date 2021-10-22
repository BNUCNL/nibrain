import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_avg_angle, s1200_avg_eccentricity
from cxy_visual_dev.lib.algo import mask_maps

work_dir = pjoin(proj_dir, 'analysis/mask_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    mask_maps(
        data_file=s1200_avg_eccentricity,
        atlas_name='MMP-vis2', roi_names=['MMP_vis2'],
        out_file=pjoin(work_dir, 'HCPY-ecc_mask-MMP-vis2.dscalar.nii')
    )

    mask_maps(
        data_file=s1200_avg_angle,
        atlas_name='MMP-vis2', roi_names=['MMP_vis2'],
        out_file=pjoin(work_dir, 'HCPY-ang_mask-MMP-vis2.dscalar.nii')
    )
