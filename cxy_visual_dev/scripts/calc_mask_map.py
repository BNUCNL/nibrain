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
    mask = atlas.get_mask(get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R'))[0]

    # mask_maps(
    #     data_file=pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus_MMP-vis3.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-MT_MMP-vis3.dscalar.nii')
    # )

    # mask_maps(
    #     data_file=pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-OccipitalPole_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-alff_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-alff_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-falff_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-falff_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-myelin_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-myelin_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-thickness_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-thickness_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-GBC1_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-GBC1_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-FC-strength1_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength1_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-GBC_cortex_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-GBC_cortex_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-FC-strength_cortex_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength_cortex_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-GBC_subcortex_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-GBC_subcortex_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'mean_map/HCPY-FC-strength_subcortex_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-FC-strength_subcortex_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'dev_trend/HCPD-myelin_age-map-mean_21-8.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPD-myelin_age-map-mean_21-8_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'dev_trend/HCPD-thickness_age-map-mean_21-8.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPD-thickness_age-map-mean_21-8_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'summary_map/HCPD-myelin_age-map-mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPD-myelin_age-map-mean_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'summary_map/HCPD-thickness_age-map-mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPD-thickness_age-map-mean_MMP-vis3.dscalar.nii')
    # )
    mask_maps(
        data_file='/nfs/m1/hcp/ACF-decay.dscalar.nii',
        mask=mask,
        out_file=pjoin(work_dir, 'HCPY-ACF-decay_MMP-vis3.dscalar.nii')
    )
