from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
split_dir = pjoin(proj_dir,
                  'analysis/s2/1080_fROI/refined_with_Kevin/grouping/split')
work_dir = pjoin(split_dir, 'tfMRI')


def split_half():
    """
    分别将左右脑，组1组2的被试随机分成两半
    组1的half1和half2分别编号为11, 12
    组2的half1和half2分别编号为21, 22
    """
    import numpy as np

    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{}.npy')
    trg_file = pjoin(split_dir, 'half_id_{}.npy')

    for hemi in ('lh', 'rh'):
        gids = np.load(gid_file.format(hemi))
        for gid in (1, 2):
            gh1_id = gid * 10 + 1
            gh2_id = gid * 10 + 2
            gid_idx_vec = gids == gid
            indices = np.where(gid_idx_vec)[0]
            n_gid = len(indices)
            gids[gid_idx_vec] = gh1_id
            half2_indices = np.random.choice(indices, int(n_gid/2), replace=False)
            gids[half2_indices] = gh2_id

        print(f'The size of group1-half1-{hemi}:', np.sum(gids == 11))
        print(f'The size of group1-half2-{hemi}:', np.sum(gids == 12))
        print(f'The size of group2-half1-{hemi}:', np.sum(gids == 21))
        print(f'The size of group2-half2-{hemi}:', np.sum(gids == 22))
        np.save(trg_file.format(hemi), gids)


def roi_stats(gh_id=11, hemi='lh'):
    import numpy as np
    import nibabel as nib
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import roi2label
    from commontool.io.io import save2nifti

    gh_id_file = pjoin(split_dir, f'half_id_{hemi}.npy')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     f'rois_v3_{hemi}.nii.gz')
    info_trg_file = pjoin(split_dir, f'rois_info_GH{gh_id}_{hemi}.pkl')
    prob_trg_file = pjoin(split_dir, f'prob_maps_GH{gh_id}_{hemi}.nii.gz')

    gh_id_idx_vec = np.load(gh_id_file) == gh_id
    rois = nib.load(roi_file).get_data().squeeze().T[gh_id_idx_vec]

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


def get_mpm(gh_id=11, hemi='lh'):
    """maximal probability map"""
    import numpy as np
    import nibabel as nib
    from commontool.io.io import save2nifti

    thr = 0.25
    prob_map_file = pjoin(split_dir, f'prob_maps_GH{gh_id}_{hemi}.nii.gz')
    out_file = pjoin(split_dir, f'MPM_GH{gh_id}_{hemi}.nii.gz')

    prob_maps = nib.load(prob_map_file).get_data()
    supra_thr_idx_arr = prob_maps > thr
    prob_maps[~supra_thr_idx_arr] = 0
    mpm = np.argmax(prob_maps, 3)
    mpm[np.any(prob_maps, 3)] += 1

    # save
    save2nifti(out_file, mpm)


if __name__ == '__main__':
    # split_half()
    # roi_stats(gh_id=11, hemi='lh')
    # roi_stats(gh_id=11, hemi='rh')
    # roi_stats(gh_id=21, hemi='lh')
    # roi_stats(gh_id=21, hemi='rh')
    # roi_stats(gh_id=12, hemi='lh')
    # roi_stats(gh_id=12, hemi='rh')
    # roi_stats(gh_id=22, hemi='lh')
    # roi_stats(gh_id=22, hemi='rh')
    # get_mpm(gh_id=11, hemi='lh')
    # get_mpm(gh_id=11, hemi='rh')
    # get_mpm(gh_id=21, hemi='lh')
    # get_mpm(gh_id=21, hemi='rh')
    get_mpm(gh_id=12, hemi='lh')
    get_mpm(gh_id=12, hemi='rh')
    get_mpm(gh_id=22, hemi='lh')
    get_mpm(gh_id=22, hemi='rh')
