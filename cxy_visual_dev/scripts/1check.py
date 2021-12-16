import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader
from cxy_visual_dev.lib.predefine import mmp_map_file, LR_count_32k,\
    L_offset_32k, L_count_32k, R_offset_32k, R_count_32k,\
    s1200_1096_myelin, s1200_1096_thickness, s1200_avg_myelin,\
    s1200_avg_thickness, All_count_32k,\
    s1200_avg_eccentricity, s1200_avg_angle, proj_dir,\
    s1200_avg_curv, s1200_1096_curv, s1200_1096_va


def check_grayordinates():
    """
    检查所用的CIFTI数据的顶点排布
    """
    # MMP atlas
    # HCPD individual surface myelin data
    # HCPD individual surface thickness data
    # HCPA individual surface myelin data
    # HCPA individual surface thickness data
    # S1200 average myelin file
    # S1200 average thickness file
    # S1200 average curvature file
    fpaths = (
        mmp_map_file,

        '/nfs/e1/HCPD/fmriresults01/'
        'HCD2133433_V1_MR/MNINonLinear/fsaverage_LR32k/'
        'HCD2133433_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',

        '/nfs/e1/HCPD/fmriresults01/'
        'HCD2133433_V1_MR/MNINonLinear/fsaverage_LR32k/'
        'HCD2133433_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii',

        '/nfs/e1/HCPA/fmriresults01/'
        'HCA7941388_V1_MR/MNINonLinear/fsaverage_LR32k/'
        'HCA7941388_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',

        '/nfs/e1/HCPA/fmriresults01/'
        'HCA7941388_V1_MR/MNINonLinear/fsaverage_LR32k/'
        'HCA7941388_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii',

        s1200_avg_myelin,

        s1200_avg_thickness,

        s1200_avg_curv
    )

    for fpath in fpaths:
        print(fpath)
        cii = nib.load(fpath)
        assert cii.shape == (1, LR_count_32k)
        idx_map = cii.header.get_index_map(1)
        brain_models = list(idx_map.brain_models)
        assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
        assert brain_models[0].index_offset == L_offset_32k
        assert brain_models[0].index_count == L_count_32k
        assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        assert brain_models[1].index_offset == R_offset_32k
        assert brain_models[1].index_count == R_count_32k

    # S1200 1096 myelin file
    # S1200 1096 thickness file
    # S1200 1096 curvature file
    # S1200 1096 vertex area file
    fpaths = (
        s1200_1096_myelin,
        s1200_1096_thickness,
        s1200_1096_curv,
        s1200_1096_va
    )
    for fpath in fpaths:
        print(fpath)
        cii = nib.load(fpath)
        assert cii.shape == (1096, LR_count_32k)
        idx_map = cii.header.get_index_map(1)
        brain_models = list(idx_map.brain_models)
        assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
        assert brain_models[0].index_offset == L_offset_32k
        assert brain_models[0].index_count == L_count_32k
        assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        assert brain_models[1].index_offset == R_offset_32k
        assert brain_models[1].index_count == R_count_32k

    # HCPD rfMRI
    # S1200_7T_Retinotopy Eccentricity
    # S1200_7T_Retinotopy Polar Angle
    # ZhouMing's PC1
    # S1200_997_tfMRI_ALLTASKS_level2
    fpaths = (
        '/nfs/e1/HCPD/fmriresults01/HCD2133433_V1_MR/'
        'MNINonLinear/Results/rfMRI_REST1_AP/'
        'rfMRI_REST1_AP_Atlas_MSMAll_hp0_clean.dtseries.nii',

        s1200_avg_eccentricity,

        s1200_avg_angle,

        pjoin(proj_dir, 'data/space/pc1.dtseries.nii'),

        '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'
        'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMAll.dscalar.nii'
    )
    for fpath in fpaths:
        print(fpath)
        cii = nib.load(fpath)
        assert cii.shape[1] == All_count_32k
        idx_map = cii.header.get_index_map(1)
        brain_models = list(idx_map.brain_models)
        assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
        assert brain_models[0].index_offset == L_offset_32k
        assert brain_models[0].index_count == L_count_32k
        assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        assert brain_models[1].index_offset == R_offset_32k
        assert brain_models[1].index_count == R_count_32k


def check_1096_sid():
    """
    检查HCPY_SubjInfo.csv文件中的1096个被试ID及其顺序和
    S1200 1096 myelin file, S1200 1096 thickness file
    是对的上的
    """
    df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv'))
    subj_ids = df['subID'].to_list()

    fpaths = (
        s1200_1096_myelin,
        s1200_1096_thickness
    )
    for fpath in fpaths:
        reader = CiftiReader(fpath)
        tmp = [int(i.split('_')[0]) for i in reader.map_names()]
        assert tmp == subj_ids


if __name__ == '__main__':
    check_grayordinates()
    check_1096_sid()
