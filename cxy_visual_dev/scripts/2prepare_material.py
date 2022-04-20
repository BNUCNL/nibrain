import numpy as np
from os.path import join as pjoin
from magicbox.io.io import GiftiReader, CiftiReader
from cxy_visual_dev.lib.predefine import proj_dir, mmp_name2label,\
    mmp_map_file, hemi2stru, hemi2Hemi


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


if __name__ == '__main__':
    get_calcarine_sulcus(hemi='lh')
    get_calcarine_sulcus(hemi='rh')
