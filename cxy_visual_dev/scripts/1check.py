

def check_grayordinates():
    """
    检查所用的CIFTI数据的顶点排布
    """
    import nibabel as nib
    from cxy_visual_dev.lib.predefine import mmp_file, LR_count_32k
    from cxy_visual_dev.lib.predefine import L_offset_32k, L_count_32k
    from cxy_visual_dev.lib.predefine import R_offset_32k, R_count_32k

    # MMP atlas
    # HCPD individual surface myelin data
    # HCPD individual surface thickness data
    # HCPA individual surface myelin data
    # HCPA individual surface thickness data
    fpaths = (mmp_file,
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
              'HCA7941388_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii')

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


if __name__ == '__main__':
    check_grayordinates()
