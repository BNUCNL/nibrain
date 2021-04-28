from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/retest')


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
        valid_subj_idx_arr = np.any(individual_roi_idx_arr, 1)
        individual_roi_idx_arr = individual_roi_idx_arr[valid_subj_idx_arr]
        test_maps_tmp = test_maps[valid_subj_idx_arr]
        retest_maps_tmp = retest_maps[valid_subj_idx_arr]
        n_subj_tmp = test_maps_tmp.shape[0]
        print(roi_name, n_subj_tmp, 'valid subjects')

        # calculate ICC for MPM
        test_series_mpm = np.mean(test_maps_tmp[:, mpm_roi_idx_arr], 1)
        retest_series_mpm = np.mean(retest_maps_tmp[:, mpm_roi_idx_arr], 1)
        mpm_iccs[:, i] = icc(test_series_mpm, retest_series_mpm, 10000, 95)

        # calculate ICC for individual
        test_series_ind = np.zeros((n_subj_tmp,))
        retest_series_ind = np.zeros((n_subj_tmp,))
        for j in range(n_subj_tmp):
            individual_roi_idx_vec = individual_roi_idx_arr[j]
            test_series_ind[j] = np.mean(test_maps_tmp[j, individual_roi_idx_vec])
            retest_series_ind[j] = np.mean(retest_maps_tmp[j, individual_roi_idx_vec])
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
    # test_retest_icc(hemi='lh')
    # test_retest_icc(hemi='rh')
    icc_plot(hemi='lh')
    icc_plot(hemi='rh')
