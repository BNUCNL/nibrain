from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                           'retest/reliability')


def calc_meas(meas_name='activ', atlas_name='MPM'):
    import time
    import numpy as np
    import pickle as pkl
    import nibabel as nib
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import roi2label, hemi2stru

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    subj_file_45 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_file_1096 = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file_test = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'activ': pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    }
    retest_dir = '/nfs/m1/hcp/retest'
    meas2file_retest = {
        'thickness': pjoin(retest_dir, '{0}/MNINonLinear/fsaverage_LR32k/'
                           '{0}.thickness_MSMAll.32k_fs_LR.dscalar.nii'),
        'myelin': pjoin(retest_dir, '{0}/MNINonLinear/fsaverage_LR32k/'
                        '{0}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'),
        'activ': pjoin(retest_dir, '{0}/MNINonLinear/Results/tfMRI_WM/'
                       'tfMRI_WM_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/'
                       'cope20.feat/zstat1.dtseries.nii')
    }
    atlas2file = {
        'MPM': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'MPM_v3_{hemi}_0.25.nii.gz'),
        'ROIv3': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                       'rois_v3_{hemi}.nii.gz')
    }

    # outputs
    out_file = pjoin(work_dir, f'{meas_name}_{atlas_name}.pkl')

    # prepare subject IDs
    subj_ids_45 = open(subj_file_45).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    subj_ids_1096 = open(subj_file_1096).read().splitlines()
    retest_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_45]
    retest_idx_in_1096 = [subj_ids_1096.index(i) for i in subj_ids_45]
    n_subj = len(subj_ids_45)

    # prepare atlas
    hemi2atlas = {}
    for hemi in hemis:
        atlas_tmp = nib.load(atlas2file[atlas_name].format(hemi=hemi))
        atlas_tmp = atlas_tmp.get_fdata().squeeze()
        if atlas_name == 'MPM':
            atlas_tmp = np.expand_dims(atlas_tmp, 0)
            hemi2atlas[hemi] = np.repeat(atlas_tmp, n_subj, 0)
        elif atlas_name == 'ROIv3':
            hemi2atlas[hemi] = atlas_tmp.T[retest_idx_in_1080]
        else:
            raise ValueError('Not supported atlas name:', atlas_name)

    # prepare test measurement maps
    if meas_name == 'activ':
        retest_idx_vec = retest_idx_in_1080
    else:
        retest_idx_vec = retest_idx_in_1096
    meas_reader_test = CiftiReader(meas2file_test[meas_name])
    hemi2maps_test = {}
    for hemi in hemis:
        hemi2maps_test[hemi] = meas_reader_test.get_data(
            hemi2stru[hemi], True)[retest_idx_vec]

    # prepare out data
    data = {}
    for hemi in hemis:
        for roi in rois:
            roi_name = roi.split('-')[0]
            data[f"{hemi}_{roi_name}_test"] = np.ones(n_subj) * np.nan
            data[f"{hemi}_{roi_name}_retest"] = np.ones(n_subj) * np.nan

    # calculate
    for subj_idx, subj_id in enumerate(subj_ids_45):
        time1 = time.time()
        meas_reader_retest = CiftiReader(meas2file_retest[
            meas_name].format(subj_id))
        for hemi in hemis:
            meas_map_test = hemi2maps_test[hemi][subj_idx]
            meas_map_retest = meas_reader_retest.get_data(
                hemi2stru[hemi], True)[0]
            atlas_map = hemi2atlas[hemi][subj_idx]
            for roi in rois:
                idx_vec = atlas_map == roi2label[roi]
                if np.any(idx_vec):
                    roi_name = roi.split('-')[0]
                    data[f"{hemi}_{roi_name}_test"][subj_idx] =\
                        np.mean(meas_map_test[idx_vec])
                    data[f"{hemi}_{roi_name}_retest"][subj_idx] =\
                        np.mean(meas_map_retest[idx_vec])
        print(f'Finished {subj_idx+1}/{n_subj}: cost {time.time()-time1} seconds')

    # save
    pkl.dump(data, open(out_file, 'wb'))


def test_retest_icc(meas_name='activ', atlas_name='MPM'):
    import pickle as pkl
    from cxy_hcp_ffa.lib.heritability import icc

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    meas_file = pjoin(work_dir, f'{meas_name}_{atlas_name}.pkl')

    # prepare
    meas_dict = pkl.load(open(meas_file, 'rb'))

    # calculate
    for hemi in hemis:
        for roi in rois:
            item = f'{hemi}_{roi}'


if __name__ == '__main__':
    calc_meas(meas_name='activ', atlas_name='MPM')
    calc_meas(meas_name='activ', atlas_name='ROIv3')
    calc_meas(meas_name='myelin', atlas_name='MPM')
    calc_meas(meas_name='myelin', atlas_name='ROIv3')
    calc_meas(meas_name='thickness', atlas_name='MPM')
    calc_meas(meas_name='thickness', atlas_name='ROIv3')
