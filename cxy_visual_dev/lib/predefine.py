import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from os.path import join as pjoin

from magicbox.io.io import CiftiReader


proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'

# >>>CIFTI brain structure
hemi2stru = {
    'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}
hemi2Hemi = {'lh': 'L', 'rh': 'R'}
# CIFTI brain structure<<<

# >>>32k_fs_LR CIFTI
All_count_32k = 91282
LR_count_32k = 59412
L_offset_32k = 0
L_count_32k = 29696
R_offset_32k = 29696
R_count_32k = 29716

# 左脑枕极的顶点号
# 左脑枕极及其1阶近邻见data/L_OccipitalPole.label
L_OccipitalPole_32k = 23908

# 右脑枕极的顶点号
# 右脑枕极及其1阶近邻见data/R_OccipitalPole.label
R_OccipitalPole_32k = 23868
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


# >>>(Wang et al, 2015) visual ROIs
wang2015_file = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
                'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
                'lr.wang2015.32k_fs_LR.dlabel.nii'
wang2015_name2label = {}
wang2015_label2name = {}
for k, v in CiftiReader(wang2015_file).label_tables()[0].items():
    if k == 0:
        continue
    wang2015_name2label[f'L_{v.label}'] = k + 25
    wang2015_name2label[f'R_{v.label}'] = k
for k, v in wang2015_name2label.items():
    wang2015_label2name[v] = k
# (Wang et al, 2015) visual ROIs<<<


# >>>S1200_7T_Retinotopy
s1200_avg_eccentricity = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
    'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
    'S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.32k_fs_LR.dscalar.nii'
s1200_avg_angle = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
    'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
    'S1200_7T_Retinotopy181.Fit1_PolarAngle_MSMAll.32k_fs_LR.dscalar.nii'
s1200_avg_anglemirror = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
    'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
    'S1200_7T_Retinotopy181.Fit1_PolarAngleMirror_MSMAll.32k_fs_LR.dscalar.nii'
s1200_avg_RFsize = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
    'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
    'S1200_7T_Retinotopy181.Fit1_ReceptiveFieldSize_MSMAll.32k_fs_LR.dscalar.nii'
s1200_avg_R2 = '/nfs/z1/HCP/HCPYA/S1200_7T_Retinotopy_Pr_9Zkk/'\
    'S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k/'\
    'S1200_7T_Retinotopy181.Fit1_R2_MSMAll.32k_fs_LR.dscalar.nii'
# S1200_7T_Retinotopy<<<


# >>>dataset
s1200_avg_dir = '/nfs/p1/public_dataset/datasets/hcp/DATA/'\
    'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1'
s1200_avg_thickness = pjoin(
    s1200_avg_dir, 'S1200.thickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_myelin = pjoin(
    s1200_avg_dir, 'S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_curv = pjoin(
    s1200_avg_dir, 'S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_thickness = pjoin(
    s1200_avg_dir, 'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_myelin = pjoin(
    s1200_avg_dir, 'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_curv = pjoin(
    s1200_avg_dir, 'S1200.All.curvature_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_va = pjoin(
    s1200_avg_dir, 'S1200.All.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii'
)
s1200_midthickness_L = pjoin(
    s1200_avg_dir, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
)
s1200_midthickness_R = pjoin(
    s1200_avg_dir, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
)
s1200_MedialWall = pjoin(
    s1200_avg_dir, 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
)

dataset_name2dir = {
    'HCPD': '/nfs/e1/HCPD',
    'HCPY': '/nfs/m1/hcp',
    'HCPA': '/nfs/e1/HCPA'
}
dataset_name2info = {
    'HCPD': pjoin(dataset_name2dir['HCPD'], 'HCPD_SubjInfo.csv'),
    'HCPY': pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv'),
    'HCPA': pjoin(dataset_name2dir['HCPA'], 'HCPA_SubjInfo.csv'),
    'HCPD_merge-6-7': pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo_merge-6-7.csv')
}
# datatset<<<


def get_rois(name):

    # >>>ColeAnticevicNetPartition
    if name == 'Cole-vis-L':
        # ColeNet中被选为视觉相关的网络的所有ROI（左脑）
        nets = ['Primary Visual', 'Secondary Visual',
                'Posterior Multimodal', 'Ventral Multimodal']
        parcel2label = get_parcel2label_by_ColeName(nets)
        rois = [i for i in parcel2label.keys() if i.startswith('L_')]

    elif name == 'Cole-vis-R':
        # ColeNet中被选为视觉相关的网络的所有ROI（右脑）
        nets = ['Primary Visual', 'Secondary Visual',
                'Posterior Multimodal', 'Ventral Multimodal']
        parcel2label = get_parcel2label_by_ColeName(nets)
        rois = [i for i in parcel2label.keys() if i.startswith('R_')]

    elif name == 'Cole-vis-L1':
        # ColeNet中被选为视觉相关的网络的所有ROI（左脑） + L_STV
        nets = ['Primary Visual', 'Secondary Visual',
                'Posterior Multimodal', 'Ventral Multimodal']
        parcel2label = get_parcel2label_by_ColeName(nets)
        rois = [i for i in parcel2label.keys() if i.startswith('L_')]
        rois.append('L_STV')
    # ColeAnticevicNetPartition<<<

    # >>>HCP-MMP1_visual-cortex2
    elif name == 'MMP-vis2-L':
        # HCP-MMP1_visual-cortex2的所有ROI（左脑）
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex2.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if np.isnan(rid):
                continue
            rois.append(f"L_{df.loc[idx, 'area_name']}")

    elif name == 'MMP-vis2-R':
        # HCP-MMP1_visual-cortex2的所有ROI（右脑）
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex2.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if np.isnan(rid):
                continue
            rois.append(f"R_{df.loc[idx, 'area_name']}")

    elif name.startswith('MMP-vis2-G'):
        # name的格式是以MMP-vis2-G开头，后面跟的是Group的编号n
        # Group n在HCP-MMP1_visual-cortex2中的的所有ROI（不区分左右）
        trg = int(name[10:])
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex2.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if rid == trg:
                rois.append(df.loc[idx, 'area_name'])
    # HCP-MMP1_visual-cortex2<<<

    # >>>HCP-MMP1_visual-cortex3
    elif name == 'MMP-vis3-L':
        # HCP-MMP1_visual-cortex3的所有ROI（左脑）
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex3.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if np.isnan(rid):
                continue
            rois.append(f"L_{df.loc[idx, 'area_name']}")

    elif name == 'MMP-vis3-R':
        # HCP-MMP1_visual-cortex3的所有ROI（右脑）
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex3.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if np.isnan(rid):
                continue
            rois.append(f"R_{df.loc[idx, 'area_name']}")

    elif name.startswith('MMP-vis3-G'):
        # name的格式是以MMP-vis3-G开头，后面跟的是Group的编号n
        # Group n在HCP-MMP1_visual-cortex3中的的所有ROI（不区分左右）
        trg = int(name[10:])
        df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex3.csv'))
        rois = []
        for idx, rid in enumerate(df['ID_in_22Region']):
            if rid == trg:
                rois.append(df.loc[idx, 'area_name'])
    # HCP-MMP1_visual-cortex3<<<

    # >>>(Wang et al, 2015) visual ROIs
    elif name == 'Wang2015-L':
        rois = [i for i in wang2015_name2label.keys() if i.startswith('L_')]

    elif name == 'Wang2015-R':
        rois = [i for i in wang2015_name2label.keys() if i.startswith('R_')]
    # (Wang et al, 2015) visual ROIs<<<

    # >>>visual path way
    elif name == 'rPath1':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_PIT', 'R_FFC', 'R_TF', 'R_PeEc']

    elif name == 'rPath2':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_V8', 'R_VVC', 'R_TF', 'R_PeEc']

    elif name == 'rPath3':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_V4', 'R_PIT', 'R_pFus-face', 'R_mFus-face', 'R_TF', 'R_PeEc']

    elif name == 'rPath4':
        rois = ['R_V1', 'R_V2', 'R_MT', 'R_STV', 'R_VIP']

    elif name == 'rPath5':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_STV', 'R_VIP']

    elif name == 'rPath6':
        rois = ['R_V1', 'R_MT', 'R_STV', 'R_VIP']

    elif name == 'rPath7':
        rois = ['R_V1', 'R_V2', 'R_V3', 'R_V3A', 'R_V7', 'R_IPS1', 'R_VIP']
    # visual path way<<<

    else:
        raise ValueError('Not supported name')

    return rois


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
        assert self.maps.shape == (1, LR_count_32k)

    def set(self, atlas_name):
        """
        Set atlas

        Args:
            atlas_name (str): atlas name
                'cortex': 左右cortex分别作为两个大ROI
                'HCP-MMP': HCP MMP1.0的所有ROI
                'FFA': MPMs of pFus- and mFus-face from my HCP-YA FFA project
                'Wang2015': (Wang et al., 2015)中定的视觉区
        """
        self.atlas_name = atlas_name

        if atlas_name == 'cortex':
            self.maps = np.ones((1, LR_count_32k), dtype=np.uint8)
            self.maps[0, R_offset_32k:(R_offset_32k+R_count_32k)] = 2
            self.roi2label = {'L_cortex': 1, 'R_cortex': 2}

        elif atlas_name == 'HCP-MMP':
            self.maps = nib.load(mmp_map_file).get_fdata()
            self.roi2label = mmp_name2label

        elif atlas_name == 'FFA':
            fsr_map = nib.load(
                pjoin(proj_dir, 'data/FFA_mpmLR32k.dlabel.nii')).get_fdata()
            self.maps = np.zeros_like(fsr_map, dtype=np.uint8)
            for lbl in ffa_name2label.values():
                self.maps[fsr_map == lbl] = lbl
            self.roi2label = ffa_name2label

        elif atlas_name == 'Wang2015':
            reader = CiftiReader(wang2015_file)
            map_L, _, _ = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT')
            map_R, _, _ = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT')
            lbl_tab = reader.label_tables()[0]
            for k in lbl_tab.keys():
                if k == 0:
                    continue
                map_L[map_L == k] = k + 25
            self.maps = np.c_[map_L, map_R]
            self.roi2label = wang2015_name2label

        else:
            raise ValueError(f'{atlas_name} is not supported at present!')
        self.n_roi = len(self.roi2label)

    def get_mask(self, roi_names):
        """
        制定mask，将roi_names指定的ROI都设置为True，其它地方为False

        Args:
            roi_names (str | strings):
                'LR': 指代所有左右脑的ROI
                'L': 指代所有左脑的ROI
                'R': 指代所有右脑的ROI
        """
        if isinstance(roi_names, str):
            roi_names = [roi_names]

        mask = np.zeros_like(self.maps, bool)
        for roi in roi_names:
            if roi == 'LR':
                for r, l in self.roi2label.items():
                    if r.startswith('L_') or r.startswith('R_'):
                        mask = np.logical_or(mask, self.maps == l)
            elif roi in ('L', 'R'):
                for r, l in self.roi2label.items():
                    if r.startswith(f'{roi}_'):
                        mask = np.logical_or(mask, self.maps == l)
            else:
                mask = np.logical_or(
                    mask, self.maps == self.roi2label[roi])

        return mask


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
