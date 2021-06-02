from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/retest')


def get_subID(meas_name='thickness'):
    import os
    import time
    import nibabel as nib

    # parameters
    par_dir = '/nfs/m1/hcp/retest'
    meas2file = {
        'thickness': pjoin(par_dir, '{0}/MNINonLinear/fsaverage_LR32k/'
                           '{0}.thickness_MSMAll.32k_fs_LR.dscalar.nii'),
        'myelin': pjoin(par_dir, '{0}/MNINonLinear/fsaverage_LR32k/'
                        '{0}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'),
        'curv': pjoin(par_dir, '{0}/MNINonLinear/fsaverage_LR32k/'
                      '{0}.curvature_MSMAll.32k_fs_LR.dscalar.nii'),
        'activ': pjoin(par_dir, '{0}/MNINonLinear/Results/tfMRI_WM/'
                       'tfMRI_WM_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/'
                       'cope20.feat/zstat1.dtseries.nii')}

    # outputs
    log_file = pjoin(work_dir, f'get_subID_log_{meas_name}')
    trg_file = pjoin(work_dir, f'subject_id_{meas_name}')

    # prepare
    folders = [i for i in os.listdir(par_dir)
               if os.path.isdir(pjoin(par_dir, i))]
    n_folder = len(folders)
    valid_ids = []
    log_writer = open(log_file, 'w')

    # calculate
    for idx, folder in enumerate(folders, 1):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(folder)
        if not os.path.exists(meas_file):
            msg = f'{meas_file} is not exist.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        try:
            nib.load(meas_file).get_fdata()
        except OSError:
            msg = f'{meas_file} meets OSError.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        valid_ids.append(folder)
        print(f'Finished: {idx}/{n_folder}, cost: {time.time()-time1} seconds')
    log_writer.close()

    # save out
    with open(trg_file, 'w') as wf:
        wf.write('\n'.join(sorted(valid_ids)))


def get_curvature_from_test(hemi='lh'):
    """
    这里就是把45个retest被试在test session里的曲率拿过来了
    """
    import nibabel as nib
    from magicbox.io.io import save2nifti

    # inputs
    src_file = pjoin(proj_dir, f'analysis/s2/{hemi}/curvature.nii.gz')
    subj_test_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_retest_file = pjoin(work_dir, 'subject_id')

    # outputs
    out_file = pjoin(work_dir, f'curvature_{hemi}_ses-test.nii.gz')

    # prepare
    curv_maps = nib.load(src_file).get_fdata()
    subj_ids_test = open(subj_test_file).read().splitlines()
    subj_ids_retest = open(subj_retest_file).read().splitlines()
    retest_indices = [subj_ids_test.index(i) for i in subj_ids_retest]

    # calculate
    curv_retest = curv_maps[..., retest_indices]

    # save
    save2nifti(out_file, curv_retest)


def get_curvature(hemi='lh'):
    """
    把45个retest被试在retest session的curvature合并成左右脑两个nifti文件，
    主要是为了我那个程序在标定个体ROI的时候读取和显示曲率。
    之前服务器上没有retest的结构数据，我想当然地认为同一个被试的沟回曲率在
    两次测量应该是一模一样的，所以在标定v1和v2版ROI的时候参照的是test session的曲率；
    这次我下了retest的结构数据，决定用retest session的曲率校对一下retest个体ROI。
    """
    import numpy as np
    from magicbox.io.io import CiftiReader, save2nifti
    from cxy_hcp_ffa.lib.predefine import hemi2stru

    # inputs
    hemis = ('lh', 'rh')
    fpath = '/nfs/m1/hcp/retest/{0}/MNINonLinear/fsaverage_LR32k/'\
            '{0}.curvature_MSMAll.32k_fs_LR.dscalar.nii'
    subj_id_file = pjoin(work_dir, 'subject_id')

    # outputs
    out_file = pjoin(work_dir, 'curvature_{hemi}.nii.gz')

    # prepare
    subj_ids = open(subj_id_file).read().splitlines()
    n_subj = len(subj_ids)
    hemi2data = {}
    for hemi in hemis:
        hemi2data[hemi] = np.zeros((32492, 1, 1, n_subj), np.float64)

    # calculate
    for subj_idx, subj_id in enumerate(subj_ids):
        reader = CiftiReader(fpath.format(subj_id))
        for hemi in hemis:
            data_tmp = reader.get_data(hemi2stru[hemi], True)[0]
            hemi2data[hemi][:, 0, 0, subj_idx] = data_tmp
        print(f'Finished: {subj_idx+1}/{n_subj}')

    # save
    for hemi in hemis:
        save2nifti(out_file.format(hemi=hemi), hemi2data[hemi])


def get_roi_idx_vec():
    """
    Get index vector with bool values for each ROI.
    The length of each index vector is matched with 45 subjects.
    True value means the ROI is delineated in the corresponding subject.
    """
    import pandas as pd

    # inputs
    src_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               'rois_v3_idx_vec.csv')
    subj_file_45 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')

    # outputs
    out_file = pjoin(work_dir, 'rois_v3_idx_vec.csv')

    # prepare
    subj_ids_45 = open(subj_file_45).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    retest_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_45]
    src_df = pd.read_csv(src_file)

    # calculate
    df = src_df.loc[retest_idx_in_1080]

    # save
    df.to_csv(out_file, index=False)


def count_roi():
    """
    Count valid subjects for each ROI
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(pjoin(work_dir, 'rois_v3_idx_vec.csv'))
    for col in df.columns:
        print(f'#subjects of {col}:', np.sum(df[col]))


def test_retest_icc(hemi='lh'):
    import numpy as np
    import pickle as pkl
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label, hemi2stru
    from cxy_hcp_ffa.lib.heritability import icc
    from magicbox.io.io import CiftiReader

    # inputs
    roi_names = ('pFus-face', 'mFus-face')
    test_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    retest_file = pjoin(work_dir, 'activation.dscalar.nii')
    subj_test_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_retest_file = pjoin(work_dir, 'subject_id')
    mpm_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'MPM_v3_{hemi}_0.25_FFA.nii.gz')
    individual_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                      f'rois_v3_{hemi}.nii.gz')

    # outputs
    out_file = pjoin(work_dir, f'ICC_{hemi}.pkl')

    # prepare
    subj_ids_test = open(subj_test_file).read().splitlines()
    subj_ids_retest = open(subj_retest_file).read().splitlines()
    retest_indices = [subj_ids_test.index(i) for i in subj_ids_retest]
    test_maps = CiftiReader(test_file).get_data(hemi2stru[hemi], True)[retest_indices]
    retest_maps = CiftiReader(retest_file).get_data(hemi2stru[hemi], True)
    mpm_mask = nib.load(mpm_file).get_fdata().squeeze()
    individual_mask = nib.load(individual_file).get_fdata().squeeze().T[retest_indices]

    # calculate
    n_roi = len(roi_names)
    mpm_iccs = np.ones((3, n_roi)) * np.nan
    individual_iccs = np.ones((3, n_roi)) * np.nan
    for i, roi_name in enumerate(roi_names):
        label = roi2label[roi_name]
        mpm_roi_idx_arr = mpm_mask == label
        individual_roi_idx_arr = individual_mask == label
        valid_subj_idx_vec = np.any(individual_roi_idx_arr, 1)
        individual_roi_idx_arr = individual_roi_idx_arr[valid_subj_idx_vec]
        test_maps_tmp = test_maps[valid_subj_idx_vec]
        retest_maps_tmp = retest_maps[valid_subj_idx_vec]
        n_subj_valid = np.sum(valid_subj_idx_vec)
        print(f'{hemi}_{roi_name}: {n_subj_valid} valid subjects')

        # calculate ICC for MPM
        test_series_mpm = np.mean(test_maps_tmp[:, mpm_roi_idx_arr], 1)
        retest_series_mpm = np.mean(retest_maps_tmp[:, mpm_roi_idx_arr], 1)
        mpm_iccs[:, i] = icc(test_series_mpm, retest_series_mpm, 10000, 95)

        # calculate ICC for individual
        test_series_ind = np.zeros((n_subj_valid,))
        retest_series_ind = np.zeros((n_subj_valid,))
        for j in range(n_subj_valid):
            individual_roi_idx_vec = individual_roi_idx_arr[j]
            test_series_ind[j] = np.mean(test_maps_tmp[j][individual_roi_idx_vec])
            retest_series_ind[j] = np.mean(retest_maps_tmp[j][individual_roi_idx_vec])
        individual_iccs[:, i] = icc(test_series_ind, retest_series_ind, 10000, 95)

    # save
    out_data = {'roi_name': roi_names, 'mpm': mpm_iccs, 'individual': individual_iccs}
    pkl.dump(out_data, open(out_file, 'wb'))


def icc_plot(hemi='lh'):
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    icc_file = pjoin(work_dir, f'ICC_{hemi}.pkl')
    data = pkl.load(open(icc_file, 'rb'))
    x = np.arange(len(data['roi_name']))
    width = auto_bar_width(x, 2)

    y_mpm = data['mpm'][1]
    low_err_mpm = y_mpm - data['mpm'][0]
    high_err_mpm = data['mpm'][2] - y_mpm
    yerr_mpm = np.array([low_err_mpm, high_err_mpm])

    y_ind = data['individual'][1]
    low_err_ind = y_ind - data['individual'][0]
    high_err_ind = data['individual'][2] - y_ind
    yerr_ind = np.array([low_err_ind, high_err_ind])

    plt.bar(x-(width/2), y_mpm, width, yerr=yerr_mpm, label='group')
    plt.bar(x+(width/2), y_ind, width, yerr=yerr_ind, label='individual')
    plt.xticks(x, data['roi_name'])
    plt.ylabel('ICC')
    plt.title(hemi)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get_subID(meas_name='thickness')
    # get_subID(meas_name='myelin')
    # get_subID(meas_name='curv')
    # get_subID(meas_name='activ')
    # get_curvature()
    get_roi_idx_vec()
    count_roi()
    # test_retest_icc(hemi='lh')
    # test_retest_icc(hemi='rh')
    # icc_plot(hemi='lh')
    # icc_plot(hemi='rh')
