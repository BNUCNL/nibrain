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


def remove_subjects(fpath=pjoin(work_dir, 'activ_MPM.pkl')):
    """
    对每个ROI，比如右脑mFus-face，从数据中去掉未标定出该ROI的被试
    """
    import os
    import pandas as pd
    import pickle as pkl

    # inputs
    idx_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               'retest/rois_v3_idx_vec.csv')

    # outputs
    fname = os.path.basename(fpath)
    fname = f"{fname.rstrip('.pkl')}_rm-subj.pkl"
    out_file = pjoin(work_dir, fname)

    # prepare
    data = pkl.load(open(fpath, 'rb'))
    idx_df = pd.read_csv(idx_file)

    # calculate
    for k, v in data.items():
        hemi_roi = '_'.join(k.split('_')[:2]) + '-face'
        data[k] = v[idx_df[hemi_roi]]

    # save
    pkl.dump(data, open(out_file, 'wb'))


def test_retest_icc(meas_name='activ', atlas_name='MPM'):
    import time
    import pickle as pkl
    from cxy_hcp_ffa.lib.heritability import icc

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    meas_file = pjoin(work_dir, f'{meas_name}_{atlas_name}_rm-subj.pkl')

    # outputs
    out_file = pjoin(work_dir, f'{meas_name}_{atlas_name}_rm-subj_icc.pkl')

    # prepare
    meas = pkl.load(open(meas_file, 'rb'))
    data = {}

    # calculate
    for hemi in hemis:
        for roi in rois:
            time1 = time.time()
            k = f'{hemi}_{roi}'
            data[k] = icc(meas[f'{k}_test'], meas[f'{k}_retest'], 10000, 95)
            print(f'Finished {k}: cost {time.time()-time1} seconds.')

    # save
    pkl.dump(data, open(out_file, 'wb'))


def test_retest_corr(meas_name='activ', atlas_name='MPM'):
    import time
    import numpy as np
    import pickle as pkl
    import nibabel as nib
    from scipy.stats.stats import pearsonr
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
    out_file = pjoin(work_dir, f'{meas_name}_{atlas_name}_corr.pkl')

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
            data[f"{hemi}_{roi_name}"] = np.ones(n_subj) * np.nan

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
                if np.sum(idx_vec) == 0:
                    pass
                elif np.sum(idx_vec) == 1:
                    print(f'{subj_id}_{hemi}_{roi} has only one vertex.')
                else:
                    r = pearsonr(meas_map_test[idx_vec],
                                 meas_map_retest[idx_vec])[0]
                    roi_name = roi.split('-')[0]
                    data[f"{hemi}_{roi_name}"][subj_idx] = r
        print(f'Finished {subj_idx+1}/{n_subj}: cost {time.time()-time1} seconds')

    # save
    pkl.dump(data, open(out_file, 'wb'))


def remove_subjects_corr(meas_name='activ'):
    import numpy as np
    import pickle as pkl

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    ind_file = pjoin(work_dir, f'{meas_name}_ROIv3_corr.pkl')
    mpm_file = pjoin(work_dir, f'{meas_name}_MPM_corr.pkl')

    # outputs
    ind_out_file = pjoin(work_dir, f'{meas_name}_ROIv3_corr_rm-subj.pkl')
    mpm_out_file = pjoin(work_dir, f'{meas_name}_MPM_corr_rm-subj.pkl')

    # prepare
    ind_data = pkl.load(open(ind_file, 'rb'))
    mpm_data = pkl.load(open(mpm_file, 'rb'))

    # calculate
    ind_out_data = {}
    mpm_out_data = {}
    for hemi in hemis:
        for roi in rois:
            k = f'{hemi}_{roi}'
            non_nan_vec = ~np.isnan(ind_data[k])
            ind_out_data[k] = ind_data[k][non_nan_vec]
            mpm_out_data[k] = mpm_data[k][non_nan_vec]

    # save
    pkl.dump(ind_out_data, open(ind_out_file, 'wb'))
    pkl.dump(mpm_out_data, open(mpm_out_file, 'wb'))


if __name__ == '__main__':
    # calc_meas(meas_name='activ', atlas_name='MPM')
    # calc_meas(meas_name='activ', atlas_name='ROIv3')
    # calc_meas(meas_name='myelin', atlas_name='MPM')
    # calc_meas(meas_name='myelin', atlas_name='ROIv3')
    # calc_meas(meas_name='thickness', atlas_name='MPM')
    # calc_meas(meas_name='thickness', atlas_name='ROIv3')
    # remove_subjects(fpath=pjoin(work_dir, 'activ_MPM.pkl'))
    # remove_subjects(fpath=pjoin(work_dir, 'activ_ROIv3.pkl'))
    # remove_subjects(fpath=pjoin(work_dir, 'myelin_MPM.pkl'))
    # remove_subjects(fpath=pjoin(work_dir, 'myelin_ROIv3.pkl'))
    # remove_subjects(fpath=pjoin(work_dir, 'thickness_MPM.pkl'))
    # remove_subjects(fpath=pjoin(work_dir, 'thickness_ROIv3.pkl'))
    # test_retest_icc(meas_name='activ', atlas_name='MPM')
    # test_retest_icc(meas_name='activ', atlas_name='ROIv3')
    # test_retest_icc(meas_name='myelin', atlas_name='MPM')
    # test_retest_icc(meas_name='myelin', atlas_name='ROIv3')
    # test_retest_icc(meas_name='thickness', atlas_name='MPM')
    # test_retest_icc(meas_name='thickness', atlas_name='ROIv3')
    # test_retest_corr(meas_name='activ', atlas_name='MPM')
    # test_retest_corr(meas_name='activ', atlas_name='ROIv3')
    # test_retest_corr(meas_name='myelin', atlas_name='MPM')
    # test_retest_corr(meas_name='myelin', atlas_name='ROIv3')
    # test_retest_corr(meas_name='thickness', atlas_name='MPM')
    # test_retest_corr(meas_name='thickness', atlas_name='ROIv3')
    remove_subjects_corr(meas_name='activ')
    remove_subjects_corr(meas_name='myelin')
    remove_subjects_corr(meas_name='thickness')
