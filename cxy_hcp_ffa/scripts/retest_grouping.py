from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                           'retest/grouping')


def FFA_config_confusion_matrix(hemi='lh'):
    import numpy as np
    import pickle as pkl
    import nibabel as nib

    # inputs
    gids = (-1, 0, 1, 2)
    configs = ('pFus', 'mFus', 'two-C', 'two-S')
    gid_file1 = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                f'grouping/group_id_{hemi}.npy')
    gid_file2 = pjoin(work_dir, f'group_id_{hemi}.npy')
    subj_file1 = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_file2 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                 'retest/subject_id')
    rois_file1 = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                 f'rois_v3_{hemi}.nii.gz')
    rois_file2 = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                 f'retest/rois_{hemi}_v2.nii.gz')

    # outputs
    out_file = pjoin(work_dir, f'FFA_config_confusion_mat_{hemi}.pkl')

    n_config = len(configs)
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

    # print('\t' + '\t'.join(configs))
    # for i, gid1 in enumerate(gids):
    #     row = [configs[i]]
    #     gid_idx_vec1 = gid_vec1 == gid1
    #     for gid2 in gids:
    #         gid_idx_vec2 = gid_vec2 == gid2
    #         gid_idx_vec = np.logical_and(gid_idx_vec1, gid_idx_vec2)
    #         row.append(str(np.sum(gid_idx_vec)))
    #     print('\t'.join(row))

    data = {'shape': 'n_test x n_retest',
            'configuration': configs,
            'matrix': np.ones((n_config, n_config), int) * -1}
    for gid1_idx, gid1 in enumerate(gids):
        gid_idx_vec1 = gid_vec1 == gid1
        for gid2_idx, gid2 in enumerate(gids):
            gid_idx_vec2 = gid_vec2 == gid2
            gid_idx_vec = np.logical_and(gid_idx_vec1, gid_idx_vec2)
            data['matrix'][gid1_idx, gid2_idx] = np.sum(gid_idx_vec)
    pkl.dump(data, open(out_file, 'wb'))


def plot_FFA_config_confusion_matrix():
    """
    样图参见figures-20210602.pptx Supplemental Figure S2
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    aspect = 2
    figsize = (6.4, 4.8)
    fpath = pjoin(work_dir, 'FFA_config_confusion_mat_{}.pkl')

    # outputs
    out_file = pjoin(work_dir, 'FFA_config_confusion_mat.jpg')

    # prepare
    data_lh = pkl.load(open(fpath.format('lh'), 'rb'))
    data_rh = pkl.load(open(fpath.format('rh'), 'rb'))
    assert data_lh['configuration'] == data_rh['configuration']
    configs = data_lh['configuration']
    n_config = len(configs)
    arr = data_lh['matrix']
    for i in range(n_config):
        arr = np.insert(arr, i*2, data_rh['matrix'][:, i], axis=1)
    ticks = np.arange(n_config)

    # plot
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, 'autumn', aspect=aspect)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.set_xticks(ticks*2+0.5)
    ax.set_xticklabels(configs)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    ax.set_yticks(ticks)
    ax.set_yticklabels(configs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(np.arange(n_config)*2-.5, minor=True)
    ax.set_yticks(np.arange(n_config)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n_config):
        for j in range(n_config*2):
            ax.text(j, i, arr[i, j],
                    ha="center", va="center", color="k")
    ax.text(-1, -0.5, 'test', ha="center", va="center", color="k")
    ax.text(-0.5, -0.75, 'retest', ha="center", va="center", color="k")
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def plot_FFA_config_confusion_matrix1():
    """
    样图参见figures-20210604.pptx Supplemental Figure S2
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    hemis = ('lh', 'rh')
    figsize = (6.4, 4.8)
    fpath = pjoin(work_dir, 'FFA_config_confusion_mat_{}.pkl')

    # outputs
    out_file = pjoin(work_dir, 'FFA_config_confusion_mat1.jpg')

    # prepare
    n_hemi = len(hemis)

    # plot
    _, axes = plt.subplots(1, n_hemi, figsize=figsize)
    for hemi_idx, hemi in enumerate(hemis):
        ax = axes[hemi_idx]

        data = pkl.load(open(fpath.format(hemi), 'rb'))
        configs = data['configuration']
        n_config = len(configs)
        ticks = np.arange(n_config)

        arr = data['matrix'] + data['matrix'].T
        diag_idx_arr = np.eye(n_config, dtype=bool)
        arr[diag_idx_arr] = arr[diag_idx_arr] / 2
        tril_mask = np.tri(n_config, k=-1)
        arr = np.ma.array(arr, mask=tril_mask)

        ax.imshow(arr, 'autumn')
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xticks(ticks)
        ax.set_xticklabels(configs)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        if hemi_idx == 0:
            ax.set_yticks(ticks)
            ax.set_yticklabels(configs)
        else:
            ax.set_yticks(ticks)
            ax.tick_params(left=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks(np.arange(n_config)-.5, minor=True)
        ax.set_yticks(np.arange(n_config)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(n_config):
            for j in range(n_config):
                ax.text(j, i, arr[i, j],
                        ha="center", va="center", color="k")
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


if __name__ == '__main__':
    # FFA_config_confusion_matrix(hemi='lh')
    # FFA_config_confusion_matrix(hemi='rh')
    # plot_FFA_config_confusion_matrix()
    plot_FFA_config_confusion_matrix1()
