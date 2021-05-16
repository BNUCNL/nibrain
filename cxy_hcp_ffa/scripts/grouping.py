from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/grouping')


def roi_stats(gid=1, hemi='lh'):
    import numpy as np
    import nibabel as nib
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import roi2label
    from magicbox.io.io import save2nifti

    # inputs
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    gid_file = pjoin(work_dir, f'group_id_{hemi}.npy')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rois_v3_{hemi}.nii.gz')

    # outputs
    rois_info_file = pjoin(work_dir, f'rois_info_{gid}_{hemi}.pkl')
    prob_maps_file = pjoin(work_dir, f'prob_maps_{gid}_{hemi}.nii.gz')

    # load
    gid_idx_vec = np.load(gid_file) == gid
    roi_maps = nib.load(roi_file).get_data().squeeze().T[gid_idx_vec]

    # prepare
    rois_info = dict()
    prob_maps = np.zeros((roi_maps.shape[1], 1, 1, len(rois)),
                         dtype=np.float64)

    # calculate
    for roi_idx, roi in enumerate(rois):
        label = roi2label[roi]
        rois_info[roi] = dict()

        # get indices of subjects which contain the roi
        indices = roi_maps == label
        subj_indices = np.any(indices, 1)

        # calculate the number of the valid subjects
        n_subject = np.sum(subj_indices)
        rois_info[roi]['n_subject'] = n_subject

        # calculate roi sizes for each valid subject
        sizes = np.sum(indices[subj_indices], 1)
        rois_info[roi]['sizes'] = sizes

        # calculate roi probability map among valid subjects
        prob_map = np.mean(indices[subj_indices], 0)
        prob_maps[:, 0, 0, roi_idx] = prob_map

    # save
    pkl.dump(rois_info, open(rois_info_file, 'wb'))
    save2nifti(prob_maps_file, prob_maps)


def plot_roi_info(gid=1, hemi='lh'):
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt
    from magicbox.algorithm.plot import show_bar_value, auto_bar_width

    roi_info_file = pjoin(work_dir, f'rois_info_{gid}_{hemi}.pkl')
    roi_infos = pkl.load(open(roi_info_file, 'rb'))

    # -plot n_subject-
    x_labels = list(roi_infos.keys())
    n_roi = len(x_labels)
    x = np.arange(n_roi)
    width = auto_bar_width(x)

    plt.figure()
    y_n_subj = [info['n_subject'] for info in roi_infos.values()]
    rects_subj = plt.bar(x, y_n_subj, width, facecolor='white', edgecolor='black')
    show_bar_value(rects_subj)
    plt.xticks(x, x_labels)
    plt.ylabel('#subject')

    # -plot sizes-
    for roi, info in roi_infos.items():
        plt.figure()
        sizes = info['sizes']
        bins = np.linspace(min(sizes), max(sizes), 40)
        _, _, patches = plt.hist(sizes, bins, color='white', edgecolor='black')
        plt.xlabel('#vertex')
        plt.title(f'distribution of {roi} sizes')
        show_bar_value(patches, '.0f')

    plt.tight_layout()
    plt.show()


def calc_prob_map_similarity():
    """
    为每个ROI概率图计算组间dice系数和pearson相关
    """
    import numpy as np
    import nibabel as nib
    from scipy.stats import pearsonr

    # inputs
    hemis = ('lh', 'rh')
    gids = (0, 1, 2)
    roi2idx = {'pFus-face': 1, 'mFus-face': 2}
    prob_file = pjoin(work_dir, 'prob_maps_{}_{}.nii.gz')

    # calculate
    for hemi in hemis:
        gid2prob_map = {}
        for gid in gids:
            gid2prob_map[gid] = nib.load(
                prob_file.format(gid, hemi)).get_fdata().squeeze().T
        for idx1, gid1 in enumerate(gids[:-1]):
            for gid2 in gids[idx1+1:]:
                for roi, prob_idx in roi2idx.items():
                    prob_map1 = gid2prob_map[gid1][prob_idx]
                    prob_map2 = gid2prob_map[gid2][prob_idx]
                    idx_map1 = prob_map1 != 0
                    idx_map2 = prob_map2 != 0
                    idx_map = np.logical_and(idx_map1, idx_map2)
                    dice = 2 * np.sum(idx_map) / \
                        (np.sum(idx_map1) + np.sum(idx_map2))
                    r = pearsonr(prob_map1[idx_map], prob_map2[idx_map])[0]
                    print(f'\n==={hemi}_G{gid1}-vs-G{gid2}_{roi}===')
                    print('Dice:', dice)
                    print('Pearson r:', r)


if __name__ == '__main__':
    # roi_stats(gid=0, hemi='lh')
    # roi_stats(gid=0, hemi='rh')
    # roi_stats(gid=1, hemi='lh')
    # roi_stats(gid=1, hemi='rh')
    # roi_stats(gid=2, hemi='lh')
    # roi_stats(gid=2, hemi='rh')
    # plot_roi_info(gid=0, hemi='lh')
    # plot_roi_info(gid=0, hemi='rh')
    # plot_roi_info(gid=1, hemi='lh')
    # plot_roi_info(gid=1, hemi='rh')
    # plot_roi_info(gid=2, hemi='lh')
    # plot_roi_info(gid=2, hemi='rh')
    calc_prob_map_similarity()
