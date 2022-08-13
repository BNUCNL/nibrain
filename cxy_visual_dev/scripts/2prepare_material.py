import numpy as np
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import GiftiReader, CiftiReader, save2cifti
from magicbox.algorithm.graph import bfs
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor
from cxy_visual_dev.lib.predefine import proj_dir, mmp_name2label,\
    mmp_map_file, hemi2stru, hemi2Hemi, s1200_midthickness_L,\
    s1200_midthickness_R, MedialWall, L_OccipitalPole_32k,\
    R_OccipitalPole_32k, L_MT_32k, R_MT_32k, get_rois, Atlas,\
    s1200_group_rsfc_mat


def get_calcarine_sulcus(hemi):
    """
    将flat相对于white geometry少掉的并且位于V1内的faces上的
    顶点视作距状沟的中心线。
    """
    Hemi = hemi2Hemi[hemi]
    gii_flat = GiftiReader('/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'
                           f'S1200.{Hemi}.flat.32k_fs_LR.surf.gii')
    gii_white = GiftiReader('/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'
                            f'S1200.{Hemi}.white_MSMAll.32k_fs_LR.surf.gii')
    mmp_reader = CiftiReader(mmp_map_file)
    out_file = pjoin(proj_dir, f'data/{Hemi}_CalcarineSulcus_split.label')

    # get splitted faces
    faces_flat = set(tuple(i) for i in gii_flat.faces)
    faces_white = set(tuple(i) for i in gii_white.faces)
    assert faces_flat.issubset(faces_white)
    faces_splitted = faces_white.difference(faces_flat)
    
    # get V1 mask
    mmp_map = mmp_reader.get_data(hemi2stru[hemi], True)[0]
    V1_mask = mmp_map == mmp_name2label[f'{Hemi}_V1']

    # get vertices
    vertices = set()
    for face in faces_splitted:
        if np.all(V1_mask[list(face)]):
            vertices.update(face)
    vertices = np.array(list(vertices))
    
    # save as .label file
    header = str(len(vertices))
    np.savetxt(out_file, vertices, fmt='%d', header=header,
               comments="#!ascii, label vertexes\n")


def get_OpMt_line(hemi):
    """
    枕极和MT的连线
    """
    Hemi = hemi2Hemi[hemi]
    hemi2surf = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    hemi2OP = {
        'lh': L_OccipitalPole_32k,
        'rh': R_OccipitalPole_32k}
    hemi2MT = {
        'lh': L_MT_32k, 'rh': R_MT_32k}
    out_file = pjoin(proj_dir, f'data/{Hemi}_OpMt.label')

    faces = GiftiReader(hemi2surf[hemi]).faces
    faces = MedialWall().remove_from_faces(hemi, faces)
    neighbors_list = get_n_ring_neighbor(faces)
    vertices = bfs(neighbors_list, hemi2OP[hemi], hemi2MT[hemi])

    # save as .label file
    header = str(len(vertices))
    np.savetxt(out_file, vertices, fmt='%d', header=header,
               comments="#!ascii, label vertexes\n")


def modify_wang2015():
    """
    从/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/
    S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/
    lr.wang2015.32k_fs_LR.dlabel.nii衍生出两个文件：
    1. wang2015.32k_fs_LR.dlabel.nii: 
    2. wang2015_4region.32k_fs_LR.dlabel.nii: 
    """
    Hemis = ['R', 'L']
    parts = ['early', 'dorsal', 'lateral', 'ventral']
    atlas = Atlas('Wang2015')
    out_data1 = atlas.maps.astype(np.uint8)
    out_data2 = np.zeros_like(atlas.maps, np.uint8)
    out_file1 = pjoin(proj_dir, 'data/wang2015/wang2015.32k_fs_LR.dlabel.nii')
    out_file2 = pjoin(proj_dir, 'data/wang2015/wang2015_4region.32k_fs_LR.dlabel.nii')
    lbl_tab1 = nib.cifti2.Cifti2LabelTable()
    lbl_tab2 = nib.cifti2.Cifti2LabelTable()
    lbl_tab1[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    lbl_tab2[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    key2 = 1
    part2rgba = {
        'early': (0, 0, 0, 1),
        'dorsal': (0, 1, 0, 1),
        'lateral': (0, 0, 1, 1),
        'ventral': (1, 0, 0, 1)}
    part2cmap = {
        'early': plt.cm.gray,
        'dorsal': plt.cm.summer,
        'lateral': plt.cm.winter,
        'ventral': plt.cm.autumn}
    for Hemi in Hemis:
        for part in parts:
            rois = get_rois(f'Wang2015-{part}-{Hemi}')

            # For wang2015.32k_fs_LR.dlabel.nii
            color_indices = np.linspace(0, 1, len(rois) * 2)
            for roi_idx, roi in enumerate(rois):
                key1 = atlas.roi2label[roi]
                rgba = part2cmap[part](color_indices[roi_idx])
                lbl_tab1[key1] = nib.cifti2.Cifti2Label(key1, roi, *rgba)

            # For wang2015_4region.32k_fs_LR.dlabel.nii
            mask = atlas.get_mask(rois)
            out_data2[mask] = key2
            lbl_tab2[key2] = nib.cifti2.Cifti2Label(key2, f'{Hemi}_{part}', *part2rgba[part])
            key2 += 1

    bms = CiftiReader(mmp_map_file).brain_models()
    save2cifti(out_file1, out_data1, bms, label_tables=[lbl_tab1])
    save2cifti(out_file2, out_data2, bms, label_tables=[lbl_tab2])


def modify_benson2018():
    """
    https://osf.io/knb5g/wiki/Data/:
    The retinotopic prior (also called the anatomical atlas of retinotopy) that
    was used in this project can be found in the analyses/fsaverage/ directory.
    This prior is essentially a version of the anatomical template reported by
    Benson et al., (2014) that has been updated using the newer HCP 181-subject
    group average data published by Benson et al., (2018).

    这些retinotopic prior已经被我下载到proj_dir下的data/benson2018/analyses/fsaverage中，
    此处就是将mgz格式转换成CIFTI格式
    """
    work_dir = pjoin(proj_dir, 'data/benson2018/analyses/fsaverage')
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    param_names = ['angle', 'eccen', 'sigma']
    # suspend


def process_HCPYA_grp_rsfc_mat():
    """
    对HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii
    中的数据进行一些处理。本次只是将z值转回r值

    References
    ----------
    (Margulies et al., 2016, SI Materials and Methods):
        Situating the default-mode network along a principal
        gradient of macroscale cortical organization
    """
    reader = CiftiReader(s1200_group_rsfc_mat)
    bms = reader.brain_models()
    vol = reader.volume
    data = reader.get_data()
    del reader
    data = np.tanh(data, dtype=np.float32)

    out_file = pjoin(proj_dir, 'data/HCP/S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_corr.dscalar.nii')
    save2cifti(out_file, data, bms, volume=vol)


if __name__ == '__main__':
    # get_calcarine_sulcus(hemi='lh')
    # get_calcarine_sulcus(hemi='rh')
    # get_OpMt_line(hemi='lh')
    # get_OpMt_line(hemi='rh')
    # modify_wang2015()
    process_HCPYA_grp_rsfc_mat()
