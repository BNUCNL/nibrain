import os
import time
import gdist
import numpy as np
import nibabel as nib
import pickle as pkl
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, GiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, get_rois, proj_dir,\
    mmp_map_file, L_offset_32k, L_count_32k, R_offset_32k, R_count_32k,\
    LR_count_32k, hemi2stru, s1200_midthickness_L, s1200_midthickness_R,\
    MedialWall, mmp_name2label

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


def calc_gdist1():
    """
    计算视觉皮层内两两顶点之间的测地距离
    """
    # find vertex numbers
    vis_name = 'MMP-vis3-R'
    rois = get_rois(vis_name)
    reader = CiftiReader(mmp_map_file)
    mask_map = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)[0]
    mask = np.zeros_like(mask_map, bool)
    for roi in rois:
        mask = np.logical_or(mask, mask_map == mmp_name2label[roi])
    vertices = np.where(mask)[0].astype(np.int32)
    n_vtx = len(vertices)

    # prepare geometry
    mw = MedialWall()
    gii = GiftiReader(s1200_midthickness_R)
    coords = gii.coords.astype(np.float64)
    faces = gii.faces.astype(np.int32)
    faces = mw.remove_from_faces('rh', faces)

    # calculate gdist
    ds = np.zeros((n_vtx, n_vtx))
    for vtx_idx in range(n_vtx):
        time1 = time.time()
        ds[vtx_idx] = gdist.compute_gdist(coords, faces, vertices[[vtx_idx]], vertices)
        print(f'Finished {vtx_idx + 1}/{n_vtx}, cost {time.time() - time1} seconds.')

    # save out
    out_dict = {'gdist': ds, 'vtx_number_in_32k_fs_R': vertices}
    out_file = pjoin(work_dir, f'gdist-between-all-pair-vtx_{vis_name}.pkl')
    pkl.dump(out_dict, open(out_file, 'wb'))


def calc_gdist2(v1_gdist_file, mt_gdist_file, out_file, mt_rank=None):
    """
    将离V1中心的距离map和离MT中心的距离map合并（取二者中较小的距离）。

    Args:
        v1_gdist_file (str): 离V1中心的距离map
        mt_gdist_file (str): 离MT中心的距离map
        mt_rank (str, optional): 离MT中心距离的初始值，Defaults to None.
            如果是None, 则从0开始。
            如果是str, 则从该字符串指定的脑区在v1_gdist_map中的平均距离开始。
    """
    Hemi2loc = {
        'L': (L_offset_32k, L_count_32k),
        'R': (R_offset_32k, R_count_32k)}
    atlas = Atlas('HCP-MMP')
    reader = CiftiReader(v1_gdist_file)

    v1_map = reader.get_data()[0]
    mt_map = nib.load(mt_gdist_file).get_fdata()[0]
    if mt_rank is not None:
        for Hemi in ('L', 'R'):
            offset, count = Hemi2loc[Hemi]
            mask = atlas.get_mask(f'{Hemi}_{mt_rank}')[0]
            init_gdist = np.mean(v1_map[mask])
            mt_map[offset:(offset+count)] += init_gdist

    data = np.zeros(LR_count_32k)
    v1_idx_map = v1_map < mt_map
    mt_idx_map = ~v1_idx_map
    data[v1_idx_map] = v1_map[v1_idx_map]
    data[mt_idx_map] = mt_map[mt_idx_map]
    data = np.expand_dims(data, 0)

    save2cifti(out_file, data, reader.brain_models())


if __name__ == '__main__':
    # calc_gdist_map_from_src(
    #     src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_CalcarineSulcus.label')),
    #     src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_CalcarineSulcus.label')),
    #     out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus.dscalar.nii')
    # )
    # calc_gdist_map_from_src(
    #     src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_CalcarineSulcus_split.label')),
    #     src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_CalcarineSulcus_split.label')),
    #     out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus-split.dscalar.nii')
    # )

    # calc_gdist_map_from_src(
    #     src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_MT.label')),
    #     src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_MT.label')),
    #     out_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii')
    # )

    # calc_gdist_map_from_src(
    #     src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_OccipitalPole.label')),
    #     src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_OccipitalPole.label')),
    #     out_file=pjoin(work_dir, 'gdist_src-OccipitalPole.dscalar.nii')
    # )

    # calc_gdist_map_from_src(
    #     src_lh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/L_OpMt.label')),
    #     src_rh=nib.freesurfer.read_label(pjoin(proj_dir, 'data/R_OpMt.label')),
    #     out_file=pjoin(work_dir, 'gdist_src-OpMt.dscalar.nii')
    # )

    # 计算每条radial line的测地距离map
    # fpath = pjoin(anal_dir, 'variation/MMP-vis3_RadialLine1-CS2_R.pkl')
    # out_file = pjoin(work_dir, 'gdist_src-MMP-vis3_RadialLine1-CS2_R.dscalar.nii')

    # lines = pkl.load(open(fpath, 'rb'))
    # n_line = len(lines)
    # out_maps = np.zeros((n_line, LR_count_32k), np.float64)
    # map_names = []
    # for idx, line in enumerate(lines):
    #     time1 = time.time()
    #     out_maps[idx] = calc_gdist_map_from_src(
    #         src_lh=None, src_rh=line)
    #     map_names.append(str(line[-1]))
    #     print(f'Finished {idx+1}/{n_line}: cost {time.time() - time1} seconds')
    # save2cifti(out_file, out_maps,
    #            CiftiReader(mmp_map_file).brain_models(), map_names)

    # calc_gdist1()

    calc_gdist2(
        v1_gdist_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus.dscalar.nii'),
        mt_gdist_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii'),
        out_file=pjoin(work_dir, 'gdist_src-Calc+MT.dscalar.nii'),
        mt_rank=None)
    calc_gdist2(
        v1_gdist_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus.dscalar.nii'),
        mt_gdist_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii'),
        out_file=pjoin(work_dir, 'gdist_src-Calc+MT=V4.dscalar.nii'),
        mt_rank='V4')
    calc_gdist2(
        v1_gdist_file=pjoin(work_dir, 'gdist_src-OccipitalPole.dscalar.nii'),
        mt_gdist_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii'),
        out_file=pjoin(work_dir, 'gdist_src-OP+MT.dscalar.nii'),
        mt_rank=None)
    calc_gdist2(
        v1_gdist_file=pjoin(work_dir, 'gdist_src-OccipitalPole.dscalar.nii'),
        mt_gdist_file=pjoin(work_dir, 'gdist_src-MT.dscalar.nii'),
        out_file=pjoin(work_dir, 'gdist_src-OP+MT=V4.dscalar.nii'),
        mt_rank='V4')
