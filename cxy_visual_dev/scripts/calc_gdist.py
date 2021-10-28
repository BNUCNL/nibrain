import os
import gdist
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, GiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, mmp_map_file,\
    L_offset_32k, L_count_32k, R_offset_32k, R_count_32k, LR_count_32k

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'gdist')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_gdist_map_from_src_32k_fs_LR(src_lh, src_rh, out_file):
    """

    Args:
        src_lh (1D array-like): left source vertices
        src_rh (1D array-like): right source vertices
    """
    hemis = ('L', 'R')
    hemi2src = {
        'L': np.asarray(src_lh, np.int32),
        'R': np.asarray(src_rh, np.int32)
    }
    hemi2stru = {'L': 'CIFTI_STRUCTURE_CORTEX_LEFT',
                 'R': 'CIFTI_STRUCTURE_CORTEX_RIGHT'}
    hemi2gii = {
        'L': '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'
             'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii',
        'R': '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'
             'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
    }
    hemi2loc = {
        'L': (L_offset_32k, L_count_32k),
        'R': (R_offset_32k, R_count_32k)
    }
    cii = CiftiReader(mmp_map_file)
    data = np.ones((1, LR_count_32k), np.float64) * np.nan
    for hemi in hemis:
        _, shape, idx2vtx = cii.get_data(hemi2stru[hemi])
        media_wall = set(range(shape[0])).difference(idx2vtx)
        gii = GiftiReader(hemi2gii[hemi])
        coords = gii.coords.astype(np.float64)
        faces = gii.faces.astype(np.int32)
        row_indices = ~np.any(np.in1d(
            faces.ravel(), media_wall).reshape(faces.shape), 1)
        faces = faces[row_indices]
        assert np.all(np.in1d(idx2vtx, faces))
        offset, count = hemi2loc[hemi]
        data[0, offset:(offset+count)] = gdist.compute_gdist(
            coords, faces, hemi2src[hemi], np.array(idx2vtx, np.int32))
    save2cifti(out_file, data, cii.brain_models())


if __name__ == '__main__':
    calc_gdist_map_from_src_32k_fs_LR(
        src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_CalcarineSulcus.label')),
        src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_CalcarineSulcus.label')),
        out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus.dscalar.nii')
    )
