import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from os.path import join as pjoin


proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'

# >>>32k_fs_LR CIFTI
LR_count_32k = 59412
L_offset_32k = 0
L_count_32k = 29696
R_offset_32k = 29696
R_count_32k = 29716
# 32k_fs_LR CIFTI<<<

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

# >>>ColeAnticevicNetPartition
cole_net_assignment_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
    'cortex_parcel_network_assignments.mat'

cole_names = ['Primary Visual', 'Secondary Visual', 'Somatomotor',
              'Cingulo-Opercular', 'Dorsal-attention', 'Language',
              'Frontoparietal', 'Auditory', 'Default', 'Posterior Multimodal',
              'Ventral Multimodal', 'Orbito-Affective']
cole_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cole_name2label = {}
cole_label2name = {}
for name, lbl in zip(cole_names, cole_labels):
    cole_name2label[name] = lbl
    cole_label2name[lbl] = name


def get_parcel2label_by_ColeName(net_names):
    """
    根据Cole Network的名字提取所有包含的parcel及其label

    Args:
        net_names (str|list): ColeNet names
            If is str, one ColeNet name.
            If is list, a list of ColeNet names.
            12 valid names: Primary Visual, Secondary Visual,
            Somatomotor, Cingulo-Opercular, Dorsal-attention,
            Language, Frontoparietal, Auditory, Default,
            Posterior Multimodal, Ventral Multimodal, Orbito-Affective
    """
    if isinstance(net_names, str):
        net_names = [net_names]
    elif isinstance(net_names, list):
        pass
    else:
        raise TypeError("Please input str or list!")
    net_labels = [cole_name2label[name] for name in net_names]

    net_lbl_vec = loadmat(cole_net_assignment_file)['netassignments'][:, 0]
    net_lbl_vec = np.r_[net_lbl_vec[180:], net_lbl_vec[:180]]

    idx_vec = np.zeros_like(net_lbl_vec, dtype=bool)
    for lbl in net_labels:
        idx_vec = np.logical_or(idx_vec, net_lbl_vec == lbl)
    mmp_labels = np.where(idx_vec)[0] + 1

    parcel2label = {}
    for lbl in mmp_labels:
        parcel2label[mmp_label2name[lbl]] = lbl

    return parcel2label
# ColeAnticevicNetPartition<<<


# >>>FFA MPM from my HCP-YA project
ffa_names = ['R_pFus-face', 'R_mFus-face', 'L_pFus-face', 'L_mFus-face']
ffa_labels = [2, 3, 5, 6]
ffa_name2label = {}
ffa_label2name = {}
for name, lbl in zip(ffa_names, ffa_labels):
    ffa_name2label[name] = lbl
    ffa_label2name[lbl] = name
# FFA MPM from my HCP-YA project<<<

# >>>dataset
s1200_avg_dir = '/nfs/p1/public_dataset/datasets/hcp/DATA/'\
    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1'
s1200_avg_thickness = pjoin(
    s1200_avg_dir, 'S1200.thickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_myelin = pjoin(
    s1200_avg_dir, 'S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_thickness = pjoin(
    s1200_avg_dir, 'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_myelin = pjoin(
    s1200_avg_dir, 'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
)

dataset_name2dir = {
    'HCPD': '/nfs/e1/HCPD',
    'HCPY': '/nfs/m1/hcp',
    'HCPA': '/nfs/e1/HCPA'
}
dataset_name2info = {
    'HCPD': pjoin(dataset_name2dir['HCPD'], 'HCPD_SubjInfo.csv'),
    'HCPY': pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv'),
    'HCPA': pjoin(dataset_name2dir['HCPA'], 'HCPA_SubjInfo.csv')
}
# datatset<<<

# >>>visual path way
rPath1 = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_PIT', 'R_FFC', 'R_TF', 'R_PeEc']
rPath2 = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V8', 'R_VVC', 'R_TF', 'R_PeEc']
rPath3 = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_PIT', 'R_pFus-face', 'R_mFus-face', 'R_TF', 'R_PeEc']
rPath4 = ['R_V1', 'R_V2', 'R_MT', 'R_STV', 'R_VIP']
rPath5 = ['R_V1', 'R_V2', 'R_V3', 'R_STV', 'R_VIP']
rPath6 = ['R_V1', 'R_MT', 'R_STV', 'R_VIP']
rPath7 = ['R_V1', 'R_V2', 'R_V3', 'R_V3A', 'R_V7', 'R_IPS1', 'R_VIP']
# visual path way<<<


class Atlas:
    """
    atlas_name: atlas name
    maps: (n_map, n_vtx) numpy array
    roi2label: key - ROI name; value - ROI label
    """

    def __init__(self, atlas_name=None):
        """
        Args:
            atlas_name (str, optional): atlas name.
                Defaults to None.
        """
        if atlas_name is None:
            self.atlas_name = None
            self.maps = None
            self.roi2label = None
            self.n_roi = None
        else:
            self.set(atlas_name)

    def set(self, atlas_name):
        """
        Set atlas

        Args:
            atlas_name (str): atlas name
                'LR': 左右脑分别作为两个大ROI
                'Cole_visual_LR': ColeNet的左右视觉相关区域分别作为两个大ROI
                'Cole_visual_ROI': ColeNet和视觉相关的各个ROI
                'HCP_MMP1': HCP MMP1.0的所有ROI
                'FFA': MPMs of pFus- and mFus-face from my HCP-YA FFA project
        """
        self.atlas_name = atlas_name

        if atlas_name == 'Cole_visual_ROI':
            mmp_map = nib.load(mmp_map_file).get_fdata()
            self.maps = np.zeros_like(mmp_map, dtype=np.uint16)
            net_names = ['Primary Visual', 'Secondary Visual',
                         'Posterior Multimodal', 'Ventral Multimodal']
            parcel2label = get_parcel2label_by_ColeName(net_names)
            for lbl in parcel2label.values():
                self.maps[mmp_map == lbl] = lbl
            self.roi2label = parcel2label

        elif atlas_name == 'LR':
            self.maps = np.ones((1, LR_count_32k), dtype=np.uint8)
            self.maps[0, R_offset_32k:(R_offset_32k+R_count_32k)] = 2
            self.roi2label = {'L_cortex': 1, 'R_cortex': 2}

        elif atlas_name == 'Cole_visual_LR':
            mmp_map = nib.load(mmp_map_file).get_fdata()
            self.maps = np.zeros_like(mmp_map, dtype=np.uint8)
            net_names = ['Primary Visual', 'Secondary Visual',
                         'Posterior Multimodal', 'Ventral Multimodal']
            parcel2label = get_parcel2label_by_ColeName(net_names)
            for roi, lbl in parcel2label.items():
                if roi.startswith('L_'):
                    self.maps[mmp_map == lbl] = 1
                elif roi.startswith('R_'):
                    self.maps[mmp_map == lbl] = 2
                else:
                    raise ValueError('parcel name must start with L_ or R_!')
            self.roi2label = {'L_cole_visual': 1, 'R_cole_visual': 2}

        elif atlas_name == 'HCP_MMP1':
            self.maps = nib.load(mmp_map_file).get_fdata()
            self.roi2label = mmp_name2label

        elif atlas_name == 'FFA':
            fsr_map = nib.load(
                pjoin(proj_dir, 'data/FFA_mpmLR32k.dlabel.nii')).get_fdata()
            self.maps = np.zeros_like(fsr_map, dtype=np.uint8)
            for lbl in ffa_name2label.values():
                self.maps[fsr_map == lbl] = lbl
            self.roi2label = ffa_name2label

        else:
            raise ValueError(f'{atlas_name} is not supported at present!')
        self.n_roi = len(self.roi2label)
