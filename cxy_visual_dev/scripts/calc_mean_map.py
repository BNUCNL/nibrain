import os
import numpy as np
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin

work_dir = pjoin(proj_dir, 'analysis/mean_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def make_mean_map(src_file, out_file):
    reader = CiftiReader(src_file)
    data = np.nanmean(reader.get_data(), 0, keepdims=True)
    save2cifti(out_file, data, reader.brain_models(), volume=reader.volume)


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
    #     src_file=s1200_1096_myelin,
    #     out_file=pjoin(work_dir, 'HCPY-myelin_mean.dscalar.nii')
    # )
    # make_mean_map(
    #     src_file=s1200_1096_thickness,
    #     out_file=pjoin(work_dir, 'HCPY-thickness_mean.dscalar.nii')
    # )
    make_mean_map(
        src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_cortex.dscalar.nii'),
        out_file=pjoin(work_dir, 'HCPY-GBC_cortex_mean.dscalar.nii')
    )
    make_mean_map(
        src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength_cortex.dscalar.nii'),
        out_file=pjoin(work_dir, 'HCPY-FC-strength_cortex_mean.dscalar.nii')
    )
    make_mean_map(
        src_file=pjoin(proj_dir, 'data/HCP/HCPY-GBC_subcortex.dscalar.nii'),
        out_file=pjoin(work_dir, 'HCPY-GBC_subcortex_mean.dscalar.nii')
    )
    make_mean_map(
        src_file=pjoin(proj_dir, 'data/HCP/HCPY-FC-strength_subcortex.dscalar.nii'),
        out_file=pjoin(work_dir, 'HCPY-FC-strength_subcortex_mean.dscalar.nii')
    )
