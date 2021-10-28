import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_avg_angle, s1200_avg_eccentricity, s1200_avg_anglemirror,\
    s1200_avg_RFsize, Atlas, get_rois, s1200_avg_curv
from cxy_visual_dev.lib.algo import mask_maps

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'mask_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')
    mask = atlas.get_mask(get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))[0]

    # mask_maps(
    #     data_file=s1200_avg_eccentricity, mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-ecc_MMP-vis2.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=s1200_avg_angle, mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-ang_MMP-vis2.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=s1200_avg_RFsize, mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-RFsize_MMP-vis2.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=s1200_avg_anglemirror, mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-AngMir_MMP-vis2.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=s1200_avg_curv, mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-curv_MMP-vis2.dscalar.nii')
    # )

    mask_maps(
        data_file=pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
        mask=mask,
        out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus_MMP-vis2.dscalar.nii')
    )
