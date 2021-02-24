from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
split_dir = pjoin(proj_dir,
                  'analysis/s2/1080_fROI/refined_with_Kevin/split')
work_dir = pjoin(split_dir, 'tfMRI')


def calc_meas_individual(hemi='lh'):
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from commontool.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    # inputs
    work_dir = pjoin(proj_dir,
                     'analysis/s2/1080_fROI/refined_with_Kevin/tfMRI')
    rois_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                      f'refined_with_Kevin/rois_v3_{hemi}.nii.gz')
    meas_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')

    # outputs
    out_file = pjoin(work_dir, f'individual_activ_{hemi}.pkl')

    rois = nib.load(rois_file).get_data().squeeze().T
    n_roi = len(roi2label)
    meas_reader = CiftiReader(meas_file)
    meas = meas_reader.get_data(hemi2stru[hemi], True)
    n_subj = meas.shape[0]

    roi_meas = {'shape': 'n_roi x n_subj', 'roi': list(roi2label.keys()),
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for roi_idx, roi in enumerate(roi_meas['roi']):
        lbl_idx_arr = rois == roi2label[roi]
        for subj_idx in range(n_subj):
            lbl_idx_vec = lbl_idx_arr[subj_idx]
            if np.any(lbl_idx_vec):
                roi_meas['meas'][roi_idx, subj_idx] = np.mean(
                    meas[subj_idx][lbl_idx_vec])
    pkl.dump(roi_meas, open(out_file, 'wb'))


def split_half():
    """
    随机将被试分成两半
    """
    import numpy as np

    n_subj = 1080
    out_file = pjoin(split_dir, 'half_id.npy')

    half_ids = np.ones(n_subj, dtype=np.uint8)
    half2_indices = np.random.choice(n_subj, int(n_subj/2), replace=False)
    half_ids[half2_indices] = 2

    print('The size of Half1:', np.sum(half_ids == 1))
    print('The size of Half2:', np.sum(half_ids == 2))
    np.save(out_file, half_ids)


def roi_stats(hid=1, hemi='lh'):
    import numpy as np
    import nibabel as nib
    import pickle as pkl

    from cxy_hcp_ffa.lib.predefine import roi2label
    from commontool.io.io import save2nifti

    hid_file = pjoin(split_dir, 'half_id.npy')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rois_v3_{hemi}.nii.gz')
    info_trg_file = pjoin(split_dir, f'rois_info_half{hid}_{hemi}.pkl')
    prob_trg_file = pjoin(split_dir, f'prob_maps_half{hid}_{hemi}.nii.gz')

    hid_idx_vec = np.load(hid_file) == hid
    rois = nib.load(roi_file).get_data().squeeze().T[hid_idx_vec]

    # prepare rois information dict
    rois_info = dict()
    for roi in roi2label.keys():
        rois_info[roi] = dict()

    prob_maps = []
    for roi, label in roi2label.items():
        # get indices of subjects which contain the roi
        indices = rois == label
        subj_indices = np.any(indices, 1)

        # calculate the number of the valid subjects
        n_subject = np.sum(subj_indices)
        rois_info[roi]['n_subject'] = n_subject

        # calculate roi sizes for each valid subject
        sizes = np.sum(indices[subj_indices], 1)
        rois_info[roi]['sizes'] = sizes

        # calculate roi probability map among valid subjects
        prob_map = np.mean(indices[subj_indices], 0)
        prob_maps.append(prob_map)
    prob_maps = np.array(prob_maps)

    # save out
    pkl.dump(rois_info, open(info_trg_file, 'wb'))
    save2nifti(prob_trg_file, prob_maps.T[:, None, None, :])


def get_mpm(hid=1, hemi='lh'):
    """maximal probability map"""
    import numpy as np
    import nibabel as nib
    from commontool.io.io import save2nifti

    thr = 0.25
    prob_file = pjoin(split_dir, f'prob_maps_half{hid}_{hemi}.nii.gz')
    trg_file = pjoin(split_dir, f'MPM_half{hid}_{hemi}.nii.gz')
    prob_maps = nib.load(prob_file).get_data()

    supra_thr_idx_arr = prob_maps > thr
    prob_maps[~supra_thr_idx_arr] = 0
    mpm = np.argmax(prob_maps, 3)
    mpm[np.any(prob_maps, 3)] += 1

    # save
    save2nifti(trg_file, mpm)


def calc_meas_MPM(hemi='lh'):
    """
    用一半被试的MPM去提取另一半被试的激活值
    如果某个被试没有某个ROI，就不提取该被试该ROI的信号

    Args:
        hemi (str, optional): hemisphere. Defaults to 'lh'.
    """
    import numpy as np
    import pickle as pkl
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label
    from commontool.io.io import CiftiReader

    hids = (1, 2)
    hid_file = pjoin(split_dir, 'half_id.npy')
    mpm_file = pjoin(split_dir, 'MPM_half{hid}_{hemi}.nii.gz')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rois_v3_{hemi}.nii.gz')
    src_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    trg_file = pjoin(work_dir, f'activ_{hemi}.pkl')

    hid_vec = np.load(hid_file)
    n_subj = len(hid_vec)
    roi_maps = nib.load(roi_file).get_data().squeeze().T
    n_roi = len(roi2label)
    meas_reader = CiftiReader(src_file)
    meas = meas_reader.get_data(hemi2stru[hemi], True)

    out_dict = {'shape': 'n_roi x n_subj',
                'roi': list(roi2label.keys()),
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for hid in hids:
        hid_idx_vec = hid_vec == hid
        mpm = nib.load(mpm_file.format(hid=hid, hemi=hemi)
                       ).get_data().squeeze()
        for roi_idx, roi in enumerate(out_dict['roi']):
            roi_idx_vec = np.any(roi_maps == roi2label[roi], 1)
            valid_idx_vec = np.logical_and(~hid_idx_vec, roi_idx_vec)
            mpm_idx_vec = mpm == roi2label[roi]
            meas_masked = meas[valid_idx_vec][:, mpm_idx_vec]
            out_dict['meas'][roi_idx][valid_idx_vec] = np.mean(meas_masked, 1)
    pkl.dump(out_dict, open(trg_file, 'wb'))


def pre_ANOVA():
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'activ_{}.pkl')
    trg_file = pjoin(work_dir, 'activ_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi), 'rb'))
        for roi in rois:
            roi_idx = data['roi'].index(roi)
            meas_vec = data['meas'][roi_idx]
            meas_vec = meas_vec[~np.isnan(meas_vec)]
            n_valid = len(meas_vec)
            out_dict['hemi'].extend([hemi] * n_valid)
            out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
            out_dict['meas'].extend(meas_vec)
            print(f'{hemi}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def pre_ANOVA_rm():
    """
    Preparation for two-way repeated-measures ANOVA
    半球x脑区
    Only the subjects who have all four ROIs will be used.
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'activ_{}.pkl')
    roi_idx_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                   'rois_v3_idx_vec.csv')

    # outputs
    trg_file = pjoin(work_dir, 'activ_preANOVA-rm.csv')

    # load data
    roi_idx_df = pd.read_csv(roi_idx_file)
    roi_cols = [f'{i}_{j}' for i in hemis for j in rois]
    roi_idx_vec = np.all(roi_idx_df.loc[:, roi_cols], 1)

    out_dict = {}
    nan_idx_vec = np.zeros_like(roi_idx_vec, np.bool)
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi), 'rb'))
        for roi in rois:
            roi_idx = data['roi'].index(roi)
            meas_vec = data['meas'][roi_idx]
            out_dict[f"{hemi}_{roi.split('-')[0]}"] = meas_vec[roi_idx_vec]
            nan_idx_vec = np.logical_or(nan_idx_vec, np.isnan(meas_vec))
    assert np.all(roi_idx_vec == ~nan_idx_vec)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


if __name__ == '__main__':
    # split_half()
    # roi_stats(hid=1, hemi='lh')
    # roi_stats(hid=1, hemi='rh')
    # roi_stats(hid=2, hemi='lh')
    # roi_stats(hid=2, hemi='rh')
    # get_mpm(hid=1, hemi='lh')
    # get_mpm(hid=1, hemi='rh')
    # get_mpm(hid=2, hemi='lh')
    # get_mpm(hid=2, hemi='rh')
    # calc_meas_MPM(hemi='lh')
    # calc_meas_MPM(hemi='rh')
    # pre_ANOVA()
    # calc_meas_individual(hemi='lh')
    # calc_meas_individual(hemi='rh')
    pre_ANOVA_rm()
