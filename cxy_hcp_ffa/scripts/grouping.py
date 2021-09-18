import numpy as np
from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/grouping')


def merge_group():
    """
    把pFus，mFus和two-C组合并为contiguous组
    """
    hemis = ('lh', 'rh')
    gid_files = pjoin(work_dir, 'group_id_{hemi}_v2.npy')
    out_files = pjoin(work_dir, 'group_id_{hemi}_v2_merged.npy')

    for hemi in hemis:
        gid_vec = np.load(gid_files.format(hemi=hemi))
        idx_vec = np.logical_or(gid_vec == -1, gid_vec == 0)
        gid_vec[idx_vec] = 1
        np.save(out_files.format(hemi=hemi), gid_vec)


def count_subject():
    """
    统计每组的人数
    """
    hemis = ('lh', 'rh')
    gids = (1, 2)
    gid_files = pjoin(work_dir, 'group_id_{hemi}_v2_merged.npy')
    # gid2name = {-1: 'pFus', 0: 'mFus', 1: 'two-C', 2: 'two-S'}
    gid2name = {1: 'continuous', 2: 'separate'}

    hemi2gid_vec = {
        'lh': np.load(gid_files.format(hemi='lh')),
        'rh': np.load(gid_files.format(hemi='rh'))
    }
    print('the number of subjects of each group:')
    for gid in gids:
        n_subjs = []
        for hemi in hemis:
            n_subjs.append(str(np.sum(hemi2gid_vec[hemi] == gid)))
        print(f"{gid2name[gid]} ({'/'.join(hemis)}): {'/'.join(n_subjs)}")


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
    import pickle as pkl
    from scipy.stats import pearsonr

    # inputs
    hemis = ('lh', 'rh')
    gids = (0, 1, 2)
    roi2idx = {'pFus-face': 1, 'mFus-face': 2}
    prob_file = pjoin(work_dir, 'prob_maps_{}_{}.nii.gz')

    # outputs
    out_file = pjoin(work_dir, 'prob_map_similarity.pkl')

    # calculate
    n_gid = len(gids)
    data = {'shape': 'n_gid x n_gid',
            'gid': gids}
    for hemi in hemis:
        gid2prob_map = {}
        for gid in gids:
            gid2prob_map[gid] = nib.load(
                prob_file.format(gid, hemi)).get_fdata().squeeze().T
        for roi, prob_idx in roi2idx.items():
            roi_dice = f"{hemi}_{roi.split('-')[0]}_dice"
            roi_corr = f"{hemi}_{roi.split('-')[0]}_corr"
            data[roi_dice] = np.zeros((n_gid, n_gid))
            data[roi_corr] = np.zeros((n_gid, n_gid))
            for idx1, gid1 in enumerate(gids):
                for idx2, gid2 in enumerate(gids):
                    prob_map1 = gid2prob_map[gid1][prob_idx]
                    prob_map2 = gid2prob_map[gid2][prob_idx]
                    idx_map1 = prob_map1 != 0
                    idx_map2 = prob_map2 != 0
                    idx_map = np.logical_and(idx_map1, idx_map2)
                    dice = 2 * np.sum(idx_map) / \
                        (np.sum(idx_map1) + np.sum(idx_map2))
                    r = pearsonr(prob_map1[idx_map], prob_map2[idx_map])[0]
                    data[roi_dice][idx1, idx2] = dice
                    data[roi_corr][idx1, idx2] = r

    pkl.dump(data, open(out_file, 'wb'))


def plot_prob_map_similarity():
    """
    References:
        1. https://blog.csdn.net/qq_27825451/article/details/105652244
        2. https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        3. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    rois = ('pFus', 'mFus')
    hemis = ('lh', 'rh')
    meas = 'corr'  # corr or dice
    data_file = pjoin(work_dir, 'prob_map_similarity.pkl')
    gid2name = {
        0: 'single',
        1: 'two-C',
        2: 'two-S'}

    # prepare
    data = pkl.load(open(data_file, 'rb'))
    n_gid = len(data['gid'])
    ticks = np.arange(n_gid)
    gid_names = [gid2name[gid] for gid in data['gid']]

    _, axes = plt.subplots(len(rois), len(hemis))
    for roi_idx, roi in enumerate(rois):
        for hemi_idx, hemi in enumerate(hemis):
            ax = axes[roi_idx, hemi_idx]
            arr = data[f'{hemi}_{roi}_{meas}']
            img = ax.imshow(arr, 'autumn')

            if roi_idx == 1:
                ax.set_xticks(ticks)
                ax.set_xticklabels(gid_names)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                if hemi_idx == 0:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(gid_names)
                else:
                    ax.tick_params(left=False, labelleft=False)
            else:
                ax.tick_params(bottom=False, labelbottom=False)
                if hemi_idx == 0:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(gid_names)
                else:
                    ax.tick_params(left=False, labelleft=False)

            for i in range(n_gid):
                for j in range(n_gid):
                    ax.text(j, i, '{:.2f}'.format(arr[i, j]),
                            ha="center", va="center", color="k")
            ax.set_title(f'{hemi}_{roi}')
    plt.show()


if __name__ == '__main__':
    merge_group()
    count_subject()
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
    # calc_prob_map_similarity()
    # plot_prob_map_similarity()
