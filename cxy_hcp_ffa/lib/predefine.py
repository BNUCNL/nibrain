import numpy as np
import pandas as pd
from os.path import join as pjoin
from magicbox.io.io import CiftiReader

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'

# >>>32k_fs_LR CIFTI
All_count_32k = 91282
LR_count_32k = 59412
L_offset_32k = 0
L_count_32k = 29696
R_offset_32k = 29696
R_count_32k = 29716
# 32k_fs_LR CIFTI<<<

# map the name of a hemisphere to CIFTI brain structure
hemi2stru = {
    'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}

# map roi names to their labels
roi2label = {
    'IOG-face': 1,
    'pFus-face': 2,
    'mFus-face': 3
}

# map roi names to colors
roi2color = {
    'IOG-face': 'red',
    'pFus-face': 'limegreen',
    'mFus-face': 'cornflowerblue'
}

# map zygosity to label
zyg2label = {'MZ': 1, 'DZ': 2}

# map Cole network name to label
# /nfs/p1/atlases/ColeAnticevicNetPartition/network_labelfile.txt
net2label_cole = {
    'Primary Visual': 1,
    'Secondary Visual': 2,
    'Somatomotor': 3,
    'Cingulo-Opercular': 4,
    'Dorsal-attention': 5,
    'Language': 6,
    'Frontoparietal': 7,
    'Auditory': 8,
    'Default': 9,
    'Posterior Multimodal': 10,
    'Ventral Multimodal': 11,
    'Orbito-Affective': 12
}


# >>>HCP MMP1.0
mmp_map_file = '/nfs/p1/atlases/multimodal_glasser/surface/'\
               'MMP_mpmLR32k.dlabel.nii'
mmp_roilbl_file = '/nfs/p1/atlases/multimodal_glasser/roilbl_mmp.csv'


def get_name_label_of_MMP():
    """
    获取HCP MMP1.0的ROI names和labels
    """
    df = pd.read_csv(mmp_roilbl_file, skiprows=1, header=None)
    n_roi = df.shape[0] * 2
    names = np.zeros(n_roi, dtype=np.object_)
    labels = np.zeros(n_roi, dtype=np.uint16)
    for idx in df.index:
        names[idx] = df.loc[idx, 1][:-4]
        names[idx+180] = df.loc[idx, 0][:-4]
        labels[idx] = idx + 1
        labels[idx+180] = idx + 181

    return names, labels


mmp_name2label = {}
mmp_label2name = {}
for name, lbl in zip(*get_name_label_of_MMP()):
    mmp_name2label[name] = lbl
    mmp_label2name[lbl] = name
# HCP MMP1.0<<<

s1200_avg_dir = '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1'
s1200_midthickness_L = pjoin(
    s1200_avg_dir, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
)
s1200_midthickness_R = pjoin(
    s1200_avg_dir, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
)
s1200_MedialWall = pjoin(
    s1200_avg_dir, 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
)


class MedialWall:
    """
    medial wall in 32k_fs_LR space
    """

    def __init__(self, method=1):
        """
        Initialize medial wall vertices

        Args:
            method (int, optional): Defaults to 1.
                1: 直接从存有MedialWall的dlabel文件中提取顶点号
                2：找到HCP MMP1.0 atlas dlabel文件中略去的顶点号
                这两种方法的结果是一致的
        """
        if method == 1:
            reader = CiftiReader(s1200_MedialWall)
            self.L_vertices = np.where(
                reader.get_data(hemi2stru['lh'])[0][0] == 1
            )[0].tolist()
            self.R_vertices = np.where(
                reader.get_data(hemi2stru['rh'])[0][0] == 1
            )[0].tolist()
        elif method == 2:
            reader = CiftiReader(mmp_map_file)
            _, L_shape, L_idx2vtx = reader.get_data(hemi2stru['lh'])
            self.L_vertices = sorted(
                set(range(L_shape[0])).difference(L_idx2vtx)
            )
            _, R_shape, R_idx2vtx = reader.get_data(hemi2stru['rh'])
            self.R_vertices = sorted(
                set(range(R_shape[0])).difference(R_idx2vtx)
            )
        else:
            raise ValueError

    def remove_from_faces(self, hemi, faces):
        """
        去除faces中的medial wall顶点

        Args:
            hemi (str): hemisphere
                lh: left hemisphere
                rh: right hemisphere
            faces (ndarray): n_face x 3
                surface mesh顶点的三角连边关系

        Returns:
            [ndarray]: 去除medial wall之后的faces
        """
        if hemi == 'lh':
            medial_wall = self.L_vertices
        elif hemi == 'rh':
            medial_wall = self.R_vertices
        else:
            raise ValueError('hemi must be lh or rh')

        row_indices = ~np.any(np.in1d(
            faces.ravel(), medial_wall).reshape(faces.shape), 1)
        faces = faces[row_indices]

        return faces
