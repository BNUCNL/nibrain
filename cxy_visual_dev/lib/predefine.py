import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
from os.path import join as pjoin
from magicbox.io.io import CiftiReader


proj_dir = '/nfs/z1/userhome/ChenXiaYu/workingdir/study/visual_dev'

meas2color = {
    'Myelination': 'cornflowerblue', 'Thickness': 'limegreen',
    'PC1': 'purple', 'PC2': 'orange'}

beh_name2abbr = {
    'CogFluidComp_Unadj': 'FluidCog', 'CogCrystalComp_Unadj': 'CrystalCog',
    'CogTotalComp_Unadj': 'TotalCog', 'PicSeq_Unadj': 'PicSeq',
    'CardSort_Unadj': 'CardSort', 'Flanker_Unadj': 'Flanker',
    'ReadEng_Unadj': 'ReadEng', 'PicVocab_Unadj': 'PicVocab',
    'ProcSpeed_Unadj': 'ProcSpeed', 'ListSort_Unadj': 'ListSort'}
# 所有属于Cognition的测量变量
cognition_cols = [
    'PicSeq_Unadj', 'PicSeq_AgeAdj', 'CardSort_Unadj',
    'CardSort_AgeAdj', 'Flanker_Unadj', 'Flanker_AgeAdj',
    'PMAT24_A_CR', 'PMAT24_A_SI', 'PMAT24_A_RTCR', 'ReadEng_Unadj',
    'ReadEng_AgeAdj', 'PicVocab_Unadj', 'PicVocab_AgeAdj',
    'ProcSpeed_Unadj', 'ProcSpeed_AgeAdj', 'DDisc_SV_1mo_200',
    'DDisc_SV_6mo_200', 'DDisc_SV_1yr_200', 'DDisc_SV_3yr_200',
    'DDisc_SV_5yr_200', 'DDisc_SV_10yr_200', 'DDisc_SV_1mo_40K',
    'DDisc_SV_6mo_40K', 'DDisc_SV_1yr_40K', 'DDisc_SV_3yr_40K',
    'DDisc_SV_5yr_40K', 'DDisc_SV_10yr_40K', 'DDisc_AUC_200',
    'DDisc_AUC_40K', 'VSPLOT_TC', 'VSPLOT_CRTE', 'VSPLOT_OFF',
    'SCPT_TP', 'SCPT_TN', 'SCPT_FP', 'SCPT_FN', 'SCPT_TPRT',
    'SCPT_SEN', 'SCPT_SPEC', 'SCPT_LRNR', 'IWRD_TOT', 'IWRD_RTC',
    'ListSort_Unadj', 'ListSort_AgeAdj', 'CogFluidComp_Unadj',
    'CogFluidComp_AgeAdj', 'CogEarlyComp_Unadj', 'CogEarlyComp_AgeAdj',
    'CogTotalComp_Unadj', 'CogTotalComp_AgeAdj',
    'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']

# 所有属于Sensory的测量变量
sensory_cols = [
    'Noise_Comp', 'Odor_Unadj', 'Odor_AgeAdj', 'PainIntens_RawScore',
    'PainInterf_Tscore', 'Taste_Unadj', 'Taste_AgeAdj', 'Color_Vision',
    'Eye', 'EVA_Num', 'EVA_Denom', 'Correction', 'Mars_Log_Score',
    'Mars_Errs', 'Mars_Final']

# 视觉相关行为类型到其测量变量的映射
vis_beh_domain2meas = {
    'Episodic Memory': ['PicSeq_Unadj', 'PicSeq_AgeAdj'],
    'Fluid Intelligence': ['PMAT24_A_CR', 'PMAT24_A_SI', 'PMAT24_A_RTCR',
                           'PMAT24_CR/RT', 'PMAT24_CR/SI'],
    'Processing Speed': ['ProcSpeed_Unadj', 'ProcSpeed_AgeAdj'],
    'Spatial Orientation': ['VSPLOT_TC', 'VSPLOT_CRTE', 'VSPLOT_OFF',
                            'VSPLOT_CR/RT', 'VSPLOT_CR/OFF'],
    'Emotion Recognition': ['ER40_CR', 'ER40_CRT', 'ER40ANG', 'ER40FEAR',
                            'ER40HAP', 'ER40NOE', 'ER40SAD', 'ER40_CR/RT'],
    'Color Vision': ['Color_Vision', 'Eye'],
    'Contrast Sensitivity': ['Mars_Log_Score', 'Mars_Errs', 'Mars_Final'],
    'Visual Acuity': ['EVA_Num', 'EVA_Denom', 'Correction'],
    'Emotion Processing (MRI)': [
        'Emotion_Task_Acc', 'Emotion_Task_Median_RT', 'Emotion_Task_CR/RT',
        'Emotion_Task_Face_Acc', 'Emotion_Task_Face_Median_RT',
        'Emotion_Task_Shape_Acc', 'Emotion_Task_Shape_Median_RT'],
    'Working Memory (MRI)': [
        'WM_Task_Acc', 'WM_Task_Median_RT', 'WM_Task_CR/RT',
        'WM_Task_2bk_Acc', 'WM_Task_2bk_Median_RT', 'WM_Task_2bk_CR/RT',
        'WM_Task_0bk_Acc', 'WM_Task_0bk_Median_RT', 'WM_Task_0bk_CR/RT',
        'WM_Task_0bk_Body_Acc', 'WM_Task_0bk_Body_Acc_Target', 'WM_Task_0bk_Body_Acc_Nontarget',
        'WM_Task_0bk_Face_Acc', 'WM_Task_0bk_Face_Acc_Target', 'WM_Task_0bk_Face_ACC_Nontarget',
        'WM_Task_0bk_Place_Acc', 'WM_Task_0bk_Place_Acc_Target', 'WM_Task_0bk_Place_Acc_Nontarget',
        'WM_Task_0bk_Tool_Acc', 'WM_Task_0bk_Tool_Acc_Target', 'WM_Task_0bk_Tool_Acc_Nontarget',
        'WM_Task_2bk_Body_Acc', 'WM_Task_2bk_Body_Acc_Target', 'WM_Task_2bk_Body_Acc_Nontarget',
        'WM_Task_2bk_Face_Acc', 'WM_Task_2bk_Face_Acc_Target', 'WM_Task_2bk_Face_Acc_Nontarget',
        'WM_Task_2bk_Place_Acc', 'WM_Task_2bk_Place_Acc_Target', 'WM_Task_2bk_Place_Acc_Nontarget',
        'WM_Task_2bk_Tool_Acc', 'WM_Task_2bk_Tool_Acc_Target', 'WM_Task_2bk_Tool_Acc_Nontarget',
        'WM_Task_0bk_Body_Median_RT', 'WM_Task_0bk_Body_Median_RT_Target', 'WM_Task_0bk_Body_Median_RT_Nontarget',
        'WM_Task_0bk_Face_Median_RT', 'WM_Task_0bk_Face_Median_RT_Target', 'WM_Task_0bk_Face_Median_RT_Nontarget',
        'WM_Task_0bk_Place_Median_RT', 'WM_Task_0bk_Place_Median_RT_Target', 'WM_Task_0bk_Place_Median_RT_Nontarget',
        'WM_Task_0bk_Tool_Median_RT', 'WM_Task_0bk_Tool_Median_RT_Target', 'WM_Task_0bk_Tool_Median_RT_Nontarget',
        'WM_Task_2bk_Body_Median_RT', 'WM_Task_2bk_Body_Median_RT_Target', 'WM_Task_2bk_Body_Median_RT_Nontarget',
        'WM_Task_2bk_Face_Median_RT', 'WM_Task_2bk_Face_Median_RT_Target', 'WM_Task_2bk_Face_Median_RT_Nontarget',
        'WM_Task_2bk_Place_Median_RT', 'WM_Task_2bk_Place_Median_RT_Target', 'WM_Task_2bk_Place_Median_RT_Nontarget',
        'WM_Task_2bk_Tool_Median_RT', 'WM_Task_2bk_Tool_Median_RT_Target', 'WM_Task_2bk_Tool_Median_RT_Nontarget']
}

# 行为正确率除以反应时的命名映射
beh_CR_div_RT_dict = {
    'PMAT24_CR/RT': ('PMAT24_A_CR', 'PMAT24_A_RTCR'),
    'PMAT24_CR/SI': ('PMAT24_A_CR', 'PMAT24_A_SI'),
    'VSPLOT_CR/RT': ('VSPLOT_TC', 'VSPLOT_CRTE'),
    'VSPLOT_CR/OFF': ('VSPLOT_TC', 'VSPLOT_OFF'),
    'ER40_CR/RT': ('ER40_CR', 'ER40_CRT'),
    'Emotion_Task_CR/RT': ('Emotion_Task_Acc', 'Emotion_Task_Median_RT'),
    'WM_Task_CR/RT': ('WM_Task_Acc', 'WM_Task_Median_RT'),
    'WM_Task_2bk_CR/RT': ('WM_Task_2bk_Acc', 'WM_Task_2bk_Median_RT'),
    'WM_Task_0bk_CR/RT': ('WM_Task_0bk_Acc', 'WM_Task_0bk_Median_RT')
}

# >>>CIFTI brain structure
hemi2stru = {
    'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}
hemi2Hemi = {'lh': 'L', 'rh': 'R'}
Hemi2stru = {
    'L': 'CIFTI_STRUCTURE_CORTEX_LEFT',
    'R': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
}
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

# 左脑MT中心顶点号
# 左脑MT中心及其1阶近邻见data/L_MT.label
L_MT_32k = 15293

# 右脑MT中心顶点号
# 右脑MT中心及其1阶近邻见data/R_MT.label
R_MT_32k = 15291
# 32k_fs_LR CIFTI<<<

# >>>HCP MMP1.0
mmp_map_file = '/nfs/z1/atlas/multimodal_glasser/surface/'\
               'MMP_mpmLR32k.dlabel.nii'
mmp_roilbl_file = '/nfs/z1/atlas/multimodal_glasser/roilbl_mmp.csv'


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
cole_net_assignment_file = '/nfs/z1/atlases/ColeAnticevicNetPartition/'\
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

# (Haak and Beckmann, 2018, Cortex)
wang2015_early = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d']
wang2015_dorsal = ['V3A', 'V3B', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']
wang2015_lateral = ['LO1', 'LO2', 'TO1', 'TO2']
wang2015_ventral = ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2']
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
s1200_avg_dir = '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1'
s1200_avg_corrThickness = pjoin(
    s1200_avg_dir, 'S1200.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_thickness = pjoin(
    s1200_avg_dir, 'S1200.thickness_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_myelin = pjoin(
    s1200_avg_dir, 'S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_avg_curv = pjoin(
    s1200_avg_dir, 'S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii'
)
s1200_1096_corrThickness = pjoin(
    s1200_avg_dir, 'S1200.All.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'
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
s1200_group_rsfc_mat = '/nfs/z1/HCP/HCPYA/HCP_S1200_1003_rfMRI_MSMAll_'\
    'groupPCA_d4500ROW_zcorr.dconn.nii'

dataset_name2dir = {
    'HCPD': '/nfs/z1/HCP/HCPD',
    'HCPY': '/nfs/z1/HCP/HCPYA',
    'HCPA': '/nfs/z1/HCP/HCPA'
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

    elif name == 'Wang2015-early-L':
        rois = [f'L_{i}' for i in wang2015_early]

    elif name == 'Wang2015-early-R':
        rois = [f'R_{i}' for i in wang2015_early]

    elif name == 'Wang2015-dorsal-L':
        rois = [f'L_{i}' for i in wang2015_dorsal]

    elif name == 'Wang2015-dorsal-R':
        rois = [f'R_{i}' for i in wang2015_dorsal]

    elif name == 'Wang2015-lateral-L':
        rois = [f'L_{i}' for i in wang2015_lateral]

    elif name == 'Wang2015-lateral-R':
        rois = [f'R_{i}' for i in wang2015_lateral]

    elif name == 'Wang2015-ventral-L':
        rois = [f'L_{i}' for i in wang2015_ventral]

    elif name == 'Wang2015-ventral-R':
        rois = [f'R_{i}' for i in wang2015_ventral]
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
    elif name == 'Hierarchy1':
        rois = ['V1', 'V2', 'V3', 'V3B', 'V4', 'MT', 'VIP', 'FST', 'TF']
    elif name == 'Hierarchy2':
        rois = ['V1', 'V2', 'V3', 'V4', 'PIT', 'VVC', 'FFC', 'TF', 'PeEc']
    elif name == 'Hierarchy3':
        rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'PIT', 'VVC', 'FFC', 'TF', 'PeEc']
    elif name == 'Hierarchy4':
        rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'PIT', 'VVC', 'FFA1', 'FFA2', 'TF', 'PeEc']
    elif name == 'Hierarchy5':
        rois = ['V1', 'V2', 'V3', 'V4', 'V8', 'PIT', 'VVC', 'pFFA', 'mFFA', 'TF', 'PeEc']
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
            map_L = reader.get_data(hemi2stru['lh'])
            map_R = reader.get_data(hemi2stru['rh'])
            lbl_tab = reader.label_tables()[0]
            for k in lbl_tab.keys():
                if k == 0:
                    continue
                map_L[map_L == k] = k + 25
            self.maps = np.c_[map_L, map_R]
            self.roi2label = wang2015_name2label

        elif atlas_name == 'MMP-vis3-EDLV':
            reader = CiftiReader(pjoin(
                proj_dir, 'data/HCP/HCP-MMP1_visual-cortex3_EDLV.dlabel.nii'
            ))
            lbl_tab = reader.label_tables()[0]
            self.roi2label = {}
            for k in lbl_tab.keys():
                if k == 0:
                    continue
                self.roi2label[lbl_tab[k].label] = k
            self.maps = reader.get_data()

        else:
            raise ValueError(f'{atlas_name} is not supported at present!')
        self.n_roi = len(self.roi2label)

    def get_mask(self, roi_names, stru_range='cortex'):
        """
        制定mask，将roi_names指定的ROI都设置为True，其它地方为False

        Args:
            roi_names (str | strings):
                'LR': 指代所有左右脑的ROI
                'L': 指代所有左脑的ROI
                'R': 指代所有右脑的ROI
            stru_range (str):
                'cortex': limited in LR_count_32k
                'grayordinate': limited in All_count_32k
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

        if stru_range == 'cortex':
            pass
        elif stru_range == 'grayordinate':
            mask = np.c_[mask, np.zeros((1, All_count_32k-LR_count_32k), bool)]
        else:
            raise ValueError('not supported stru_range:', stru_range)
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
                reader.get_data(hemi2stru['lh'])[0] == 1
            )[0].tolist()
            self.R_vertices = np.where(
                reader.get_data(hemi2stru['rh'])[0] == 1
            )[0].tolist()
        elif method == 2:
            reader = CiftiReader(mmp_map_file)
            _, _, L_shape, L_idx2vtx = reader.get_stru_pos(hemi2stru['lh'])
            self.L_vertices = sorted(
                set(range(L_shape[0])).difference(L_idx2vtx)
            )
            _, _, R_shape, R_idx2vtx = reader.get_stru_pos(hemi2stru['rh'])
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
