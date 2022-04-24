import numpy as np
from os.path import join as pjoin
from magicbox.io.io import GiftiReader, CiftiReader
from magicbox.algorithm.graph import bfs
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor
from cxy_visual_dev.lib.predefine import proj_dir, mmp_name2label,\
    mmp_map_file, hemi2stru, hemi2Hemi, s1200_midthickness_L,\
    s1200_midthickness_R, MedialWall, L_OccipitalPole_32k,\
    R_OccipitalPole_32k, L_MT_32k, R_MT_32k


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


if __name__ == '__main__':
    # get_calcarine_sulcus(hemi='lh')
    # get_calcarine_sulcus(hemi='rh')
    get_OpMt_line(hemi='lh')
    get_OpMt_line(hemi='rh')
