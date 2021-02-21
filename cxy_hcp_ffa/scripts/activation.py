from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
split_dir = pjoin(proj_dir,
                  'analysis/s2/1080_fROI/refined_with_Kevin/split')
work_dir = pjoin(split_dir, 'tfMRI')


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
    trg_file = pjoin(split_dir, f'MPM_half{hid}_{hemi}_new.nii.gz')
    prob_maps = nib.load(prob_file).get_data()

    supra_thr_idx_arr = prob_maps > thr
    prob_maps[~supra_thr_idx_arr] = 0
    mpm = np.argmax(prob_maps, 3)
    mpm[np.any(prob_maps, 3)] += 1

    # save
    save2nifti(trg_file, mpm)


if __name__ == '__main__':
    # split_half()
    # roi_stats(hid=1, hemi='lh')
    # roi_stats(hid=1, hemi='rh')
    # roi_stats(hid=2, hemi='lh')
    # roi_stats(hid=2, hemi='rh')
    get_mpm(hid=1, hemi='lh')
    get_mpm(hid=1, hemi='rh')
    get_mpm(hid=2, hemi='lh')
    get_mpm(hid=2, hemi='rh')
