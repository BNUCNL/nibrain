from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/retest')


def get_curvature(hemi='lh'):
    import nibabel as nib
    from magicbox.io.io import save2nifti

    # inputs
    src_file = pjoin(proj_dir, f'analysis/s2/{hemi}/curvature.nii.gz')
    subj_test_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_retest_file = pjoin(work_dir, 'subject_id')

    # outputs
    out_file = pjoin(work_dir, f'curvature_{hemi}.nii.gz')

    # prepare
    curv_maps = nib.load(src_file).get_fdata()
    subj_ids_test = open(subj_test_file).read().splitlines()
    subj_ids_retest = open(subj_retest_file).read().splitlines()
    retest_indices = [subj_ids_test.index(i) for i in subj_ids_retest]

    # calculate
    curv_retest = curv_maps[..., retest_indices]

    # save
    save2nifti(out_file, curv_retest)


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


def FFA_config_confusion_matrix(hemi='lh'):
    import numpy as np
    import nibabel as nib

    gids = (-1, 0, 1, 2)
    configs = ('pFus-only', 'mFus-only', 'two-connected', 'two-separate')
    gid_file1 = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                f'grouping/group_id_{hemi}.npy')
    gid_file2 = pjoin(work_dir, f'grouping/group_id_{hemi}.npy')
    subj_file1 = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_file2 = pjoin(work_dir, 'subject_id')
    rois_file1 = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                 f'rois_v3_{hemi}.nii.gz')
    rois_file2 = pjoin(work_dir, f'rois_{hemi}_v2.nii.gz')

    subj_ids1 = open(subj_file1).read().splitlines()
    subj_ids2 = open(subj_file2).read().splitlines()
    subj_indices = [subj_ids1.index(i) for i in subj_ids2]
    gid_vec1 = np.load(gid_file1)[subj_indices]
    gid_vec2 = np.load(gid_file2)
    roi_maps1 = nib.load(rois_file1).get_fdata().squeeze().T[subj_indices]
    roi_maps2 = nib.load(rois_file2).get_fdata().squeeze().T

    for i, gid in enumerate(gid_vec1):
        if gid == 0:
            roi_labels = set(roi_maps1[i])
            if 2 in roi_labels and 3 in roi_labels:
                raise ValueError("impossible1")
            elif 2 in roi_labels:
                gid_vec1[i] = -1
            elif 3 in roi_labels:
                pass
            else:
                raise ValueError("impossible2")

    for i, gid in enumerate(gid_vec2):
        if gid == 0:
            roi_labels = set(roi_maps2[i])
            if 2 in roi_labels and 3 in roi_labels:
                raise ValueError("impossible3")
            elif 2 in roi_labels:
                gid_vec2[i] = -1
            elif 3 in roi_labels:
                pass
            else:
                raise ValueError("impossible4")

    print('\t' + '\t'.join(configs))
    for i, gid1 in enumerate(gids):
        row = [configs[i]]
        for gid2 in gids:
            gid_idx_vec1 = gid_vec1 == gid1
            gid_idx_vec2 = gid_vec2 == gid2
            gid_idx_vec = np.logical_and(gid_idx_vec1, gid_idx_vec2)
            row.append(str(np.sum(gid_idx_vec)))
        print('\t'.join(row))


if __name__ == '__main__':
    # get_curvature(hemi='lh')
    # get_curvature(hemi='rh')
    # test_retest_icc(hemi='lh')
    # test_retest_icc(hemi='rh')
    # icc_plot(hemi='lh')
    # icc_plot(hemi='rh')
    FFA_config_confusion_matrix(hemi='lh')
    FFA_config_confusion_matrix(hemi='rh')
