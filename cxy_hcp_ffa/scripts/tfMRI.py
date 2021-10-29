from os.path import join as pjoin
from cxy_hcp_ffa.lib.predefine import proj_dir
from cxy_hcp_ffa.lib.algo import meas_pkl2csv

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'tfMRI')


def calc_meas_individual(hemi='lh'):
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    # inputs
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


def calc_meas_emotion(hemi='lh'):
    """
    用个体ROI去提取emotion任务中 faces-shapes的信号
    """
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import roi2label

    # inputs
    rois_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                      f'refined_with_Kevin/rois_v3_{hemi}.nii.gz')
    subj_file_wm = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_file = pjoin(proj_dir, 'data/HCP/emotion/analysis_s2/'
                                f'cope3_face-shape_zstat_{hemi}.nii.gz')
    subj_file_emo = pjoin(proj_dir, 'data/HCP/emotion/analysis_s2/subject_id')

    # outputs
    out_file = pjoin(work_dir, f'individual_activ_{hemi}_emo.pkl')

    # prepare
    rois = nib.load(rois_file).get_fdata().squeeze().T
    n_roi = len(roi2label)
    subj_ids_wm = open(subj_file_wm).read().splitlines()
    meas = nib.load(meas_file).get_fdata().squeeze().T
    subj_ids_emo = open(subj_file_emo).read().splitlines()

    # find index in EMOTION subject IDs for WM subject IDs
    subj_indices = []
    for subj_id in subj_ids_wm:
        if subj_id in subj_ids_emo:
            subj_indices.append(subj_ids_emo.index(subj_id))
        else:
            subj_indices.append(None)
    n_subj = len(subj_indices)

    roi_meas = {'shape': 'n_roi x n_subj', 'roi': list(roi2label.keys()),
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for roi_idx, roi in enumerate(roi_meas['roi']):
        lbl_idx_arr = rois == roi2label[roi]
        for subj_idx in range(n_subj):
            lbl_idx_vec = lbl_idx_arr[subj_idx]
            subj_idx_emo = subj_indices[subj_idx]
            if np.any(lbl_idx_vec) and subj_idx_emo is not None:
                roi_meas['meas'][roi_idx, subj_idx] = np.mean(
                    meas[subj_idx_emo][lbl_idx_vec])
    pkl.dump(roi_meas, open(out_file, 'wb'))


if __name__ == '__main__':
    # calc_meas_individual(hemi='lh')
    # calc_meas_individual(hemi='rh')
    # calc_meas_emotion(hemi='lh')
    # calc_meas_emotion(hemi='rh')

    meas_pkl2csv(
        lh_file=pjoin(work_dir, 'individual_activ_lh.pkl'),
        rh_file=pjoin(work_dir, 'individual_activ_rh.pkl'),
        out_file=pjoin(work_dir, 'FFA_activ.csv'),
        rois=('pFus-face', 'mFus-face')
    )
    meas_pkl2csv(
        lh_file=pjoin(work_dir, 'individual_activ_lh_emo.pkl'),
        rh_file=pjoin(work_dir, 'individual_activ_rh_emo.pkl'),
        out_file=pjoin(work_dir, 'FFA_activ-emo.csv'),
        rois=('pFus-face', 'mFus-face')
    )
