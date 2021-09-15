import numpy as np
import pandas as pd

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'

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
