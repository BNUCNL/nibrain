import os
import gdist
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, GiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, mmp_map_file,\
    L_offset_32k, L_count_32k, R_offset_32k, R_count_32k, LR_count_32k,\
    hemi2stru, s1200_midthickness_L, s1200_midthickness_R, MedialWall

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'gdist')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_gdist_map_from_src(src_lh, src_rh, out_file=None):
    """
    基于32k_fs_LR mesh，计算所有顶点到src的测地距离

    Args:
        src_lh (1D array-like | None): left source vertices
        src_rh (1D array-like | None): right source vertices
        out_file (str, optional):
            If is str: save gdist map out
            If is None: return gdist map
    """
    hemis = ('lh', 'rh')
    hemi2src = {'lh': src_lh, 'rh': src_rh}
    hemi2gii = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    hemi2loc = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}

    mw = MedialWall()
    reader = CiftiReader(mmp_map_file)
    data = np.ones(LR_count_32k, np.float64) * np.nan
    for hemi in hemis:

        if hemi2src[hemi] is None:
            continue

        src_vertices = np.asarray(hemi2src[hemi], np.int32)
        trg_vertices = reader.get_data(hemi2stru[hemi])[-1]
        trg_vertices = np.array(trg_vertices, np.int32)

        gii = GiftiReader(hemi2gii[hemi])
        coords = gii.coords.astype(np.float64)
        faces = gii.faces.astype(np.int32)
        faces = mw.remove_from_faces(hemi, faces)

        offset, count = hemi2loc[hemi]
        data[offset:(offset+count)] = gdist.compute_gdist(
            coords, faces, src_vertices, trg_vertices)

    if out_file is None:
        return data
    else:
        data = np.expand_dims(data, 0)
        save2cifti(out_file, data, reader.brain_models())


if __name__ == '__main__':
    calc_gdist_map_from_src(
        src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_CalcarineSulcus.label')),
        src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_CalcarineSulcus.label')),
        out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus.dscalar.nii')
    )

    calc_gdist_map_from_src(
        src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_MT.label')),
        src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_MT.label')),
        out_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii')
    )

    calc_gdist_map_from_src(
        src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_OccipitalPole.label')),
        src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_OccipitalPole.label')),
        out_file=pjoin(work_dir, 'gdist_src-OccipitalPole.dscalar.nii')
    )
