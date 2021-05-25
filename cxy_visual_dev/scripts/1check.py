

def check_grayordinates():
    """
    检查所用的CIFTI数据的顶点排布
    """
    import nibabel as nib
    from cxy_visual_dev.lib.predefine import mmp_file

    n_vtx = 59412

    # MMP atlas
    fpath = mmp_file
    cii = nib.load(fpath)
    assert cii.shape == (1, n_vtx)
    idx_map = cii.header.get_index_map(1)
    brain_models = list(idx_map.brain_models)
    assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert brain_models[0].index_offset == 0
    assert brain_models[0].index_count == 29696
    assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert brain_models[1].index_offset == 29696
    assert brain_models[1].index_count == 29716

    # HCPD individual surface myelin data
    fpath = '/nfs/e1/HCPD/fmriresults01/'\
            'HCD2133433_V1_MR/MNINonLinear/fsaverage_LR32k/'\
            'HCD2133433_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
    cii = nib.load(fpath)
    assert cii.shape == (1, n_vtx)
    idx_map = cii.header.get_index_map(1)
    brain_models = list(idx_map.brain_models)
    assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert brain_models[0].index_offset == 0
    assert brain_models[0].index_count == 29696
    assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert brain_models[1].index_offset == 29696
    assert brain_models[1].index_count == 29716

    # HCPD individual surface thickness data
    fpath = '/nfs/e1/HCPD/fmriresults01/'\
            'HCD2133433_V1_MR/MNINonLinear/fsaverage_LR32k/'\
            'HCD2133433_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    cii = nib.load(fpath)
    assert cii.shape == (1, n_vtx)
    idx_map = cii.header.get_index_map(1)
    brain_models = list(idx_map.brain_models)
    assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert brain_models[0].index_offset == 0
    assert brain_models[0].index_count == 29696
    assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert brain_models[1].index_offset == 29696
    assert brain_models[1].index_count == 29716

    # HCPA individual surface myelin data
    fpath = '/nfs/e1/HCPA/fmriresults01/'\
            'HCA7941388_V1_MR/MNINonLinear/fsaverage_LR32k/'\
            'HCA7941388_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
    cii = nib.load(fpath)
    assert cii.shape == (1, n_vtx)
    idx_map = cii.header.get_index_map(1)
    brain_models = list(idx_map.brain_models)
    assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert brain_models[0].index_offset == 0
    assert brain_models[0].index_count == 29696
    assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert brain_models[1].index_offset == 29696
    assert brain_models[1].index_count == 29716

    # HCPA individual surface thickness data
    fpath = '/nfs/e1/HCPA/fmriresults01/'\
            'HCA7941388_V1_MR/MNINonLinear/fsaverage_LR32k/'\
            'HCA7941388_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    cii = nib.load(fpath)
    assert cii.shape == (1, n_vtx)
    idx_map = cii.header.get_index_map(1)
    brain_models = list(idx_map.brain_models)
    assert brain_models[0].brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert brain_models[0].index_offset == 0
    assert brain_models[0].index_count == 29696
    assert brain_models[1].brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert brain_models[1].index_offset == 29696
    assert brain_models[1].index_count == 29716


if __name__ == '__main__':
    check_grayordinates()
