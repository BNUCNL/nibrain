from os.path import join as pjoin
from matplotlib import pyplot as plt

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/paper_fig')


def plot_development():
    import numpy as np
    import pandas as pd
    import pickle as pkl
    from scipy.stats.stats import sem
    from cxy_hcp_ffa.lib.predefine import roi2color

    # inputs
    figsize = (4, 6)
    rois = ('pFus-face', 'mFus-face')
    hemis = ('lh', 'rh')
    hemi2style = {'lh': '-', 'rh': '--'}
    dev_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                              'development')
    meas2file = {
        'thickness': pjoin(dev_dir, 'HCPD_thickness_MPM1_prep_inf.csv'),
        'myelin': pjoin(dev_dir, 'HCPD_myelin_MPM1_prep_inf.csv'),
        'rsfc': pjoin(dev_dir, 'rfMRI/rsfc_MPM2Cole_{hemi}.pkl'),
    }
    meas2ylabel = {
        'thickness': 'thickness',
        'myelin': 'myelination',
        'rsfc': 'RSFC'
    }

    # outputs
    out_file = pjoin(work_dir, 'dev_line.jpg')

    # prepare
    n_meas = len(meas2file)
    age_name = 'age in years'

    # plot
    _, axes = plt.subplots(n_meas, 1, figsize=figsize)
    for meas_idx, meas_name in enumerate(meas2file.keys()):
        ax = axes[meas_idx]
        fpath = meas2file[meas_name]
        if meas_name == 'rsfc':
            subj_info = pd.read_csv(pjoin(dev_dir, 'HCPD_SubjInfo.csv'))
            age_vec = np.array(subj_info[age_name])
            for hemi in hemis:
                rsfc_dict = pkl.load(open(fpath.format(hemi=hemi), 'rb'))
                for roi in rois:
                    fc_vec = np.mean(rsfc_dict[roi], 1)
                    non_nan_vec = ~np.isnan(fc_vec)
                    fcs = fc_vec[non_nan_vec]
                    ages = age_vec[non_nan_vec]
                    age_uniq = np.unique(ages)
                    ys = np.zeros_like(age_uniq, np.float64)
                    yerrs = np.zeros_like(age_uniq, np.float64)
                    for idx, age in enumerate(age_uniq):
                        sample = fcs[ages == age]
                        ys[idx] = np.mean(sample)
                        yerrs[idx] = sem(sample)
                    ax.errorbar(age_uniq, ys, yerrs, label=f"{hemi}_{roi.split('-')[0]}",
                                color=roi2color[roi], linestyle=hemi2style[hemi])
            ax.set_ylabel(meas2ylabel[meas_name])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.legend()
            x_ticks = np.unique(age_vec)[::2]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        else:
            data = pd.read_csv(fpath)
            age_vec = np.array(data[age_name])
            age_uniq = np.unique(age_vec)
            for hemi in hemis:
                for roi in rois:
                    roi_name = roi.split('-')[0]
                    col = f"{roi_name}_{hemi}"
                    meas_vec = np.array(data[col])
                    ys = np.zeros_like(age_uniq, np.float64)
                    yerrs = np.zeros_like(age_uniq, np.float64)
                    for age_idx, age in enumerate(age_uniq):
                        sample = meas_vec[age_vec == age]
                        ys[age_idx] = np.mean(sample)
                        yerrs[age_idx] = sem(sample)
                    ax.errorbar(age_uniq, ys, yerrs, label=f'{hemi}_{roi_name}',
                                color=roi2color[roi], linestyle=hemi2style[hemi])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.legend()
            ax.set_ylabel(meas2ylabel[meas_name])
            x_ticks = age_uniq[::2]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        if meas_idx+1 == n_meas:
            ax.set_xlabel(age_name)
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def plot_development_pattern_corr():
    import numpy as np
    import pandas as pd
    from scipy.stats.stats import sem
    from cxy_hcp_ffa.lib.predefine import roi2color

    # inputs
    figsize = (4, 6)
    rois = ('pFus-face', 'mFus-face')
    hemis = ('lh', 'rh')
    hemi2style = {'lh': '-', 'rh': '--'}
    dev_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                              'development')
    subj_info_file = pjoin(dev_dir, 'HCPD_SubjInfo.csv')
    meas2file = {
        'thickness': pjoin(dev_dir, 'HCPD_thickness-corr_MPM1.csv'),
        'myelin': pjoin(dev_dir, 'HCPD_myelin-corr_MPM1.csv'),
        'rsfc': pjoin(dev_dir, 'rfMRI/rsfc-corr_MPM.csv'),
    }
    meas2title = {
        'thickness': 'thickness',
        'myelin': 'myelination',
        'rsfc': 'RSFC'
    }

    # outputs
    out_file = pjoin(work_dir, 'dev-corr_line.jpg')

    # prepare
    age_name = 'age in years'
    subj_info = pd.read_csv(subj_info_file)
    subj_ids = subj_info['subID'].to_list()
    ages = np.array(subj_info[age_name])
    n_meas = len(meas2file)

    # plot
    _, axes = plt.subplots(n_meas, 1, figsize=figsize)
    for meas_idx, meas_name in enumerate(meas2file.keys()):
        ax = axes[meas_idx]
        data = pd.read_csv(meas2file[meas_name])
        assert subj_ids == data['subID'].to_list()
        for hemi in hemis:
            for roi in rois:
                roi_name = roi.split('-')[0]
                col = f"{roi_name}_{hemi}"
                meas_vec = np.array(data[col])
                non_nan_vec = ~np.isnan(meas_vec)
                meas_vec = meas_vec[non_nan_vec]
                age_vec = ages[non_nan_vec]
                print(f'{meas_name}_{hemi}_{roi}:', meas_vec.shape)
                age_uniq = np.unique(age_vec)
                ys = np.zeros_like(age_uniq, np.float64)
                yerrs = np.zeros_like(age_uniq, np.float64)
                for age_idx, age in enumerate(age_uniq):
                    sample = meas_vec[age_vec == age]
                    ys[age_idx] = np.mean(sample)
                    yerrs[age_idx] = sem(sample)
                ax.errorbar(age_uniq, ys, yerrs, label=f'{hemi}_{roi_name}',
                            color=roi2color[roi], linestyle=hemi2style[hemi])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        if meas_name == 'thickness':
            ax.set_ylim(-0.3, 0.6)
        if meas_idx+1 == n_meas:
            ax.set_xlabel(age_name)
        # ax.set_title(meas2title[meas_name])
        ax.set_ylabel('pearson R')
        x_ticks = np.unique(ages)[::2]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def plot_prob_map_similarity():
    """
    样图参见 figures-20210602.pptx Supplemental Figure S3 heatmap
    References:
        1. https://blog.csdn.net/qq_27825451/article/details/105652244
        2. https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        3. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    figsize = (3, 3)
    aspect = 2
    rois = ('pFus', 'mFus')
    meas = 'corr'  # corr or dice
    data_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                'grouping/prob_map_similarity.pkl')
    gid2name = {
        0: 'single',
        1: 'two-C',
        2: 'two-S'}

    # outputs
    out_file = pjoin(work_dir, 'prob_similarity_{}.jpg')

    # prepare
    data = pkl.load(open(data_file, 'rb'))
    n_gid = len(data['gid'])
    ticks = np.arange(n_gid)
    gid_names = [gid2name[gid] for gid in data['gid']]

    for roi in rois:
        _, ax = plt.subplots(figsize=figsize)
        k_lh = f'lh_{roi}_{meas}'
        k_rh = f'rh_{roi}_{meas}'
        arr = data[k_lh]
        for i in range(n_gid):
            j = 2 * i
            arr = np.insert(arr, j, data[k_rh][:, i], axis=1)
            arr[i, j] = 0
            arr[i, j+1] = 0
        ax.imshow(arr, 'autumn', aspect=aspect)

        ax.tick_params(top=True, bottom=False,
                      labeltop=True, labelbottom=False)
        ax.set_xticks(ticks * 2 + 0.5)
        ax.set_xticklabels(gid_names)
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        ax.set_yticks(ticks)
        ax.set_yticklabels(gid_names)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks(np.arange(n_gid)*2-.5, minor=True)
        ax.set_yticks(np.arange(n_gid)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(n_gid):
            for j in range(n_gid*2):
                if j == 2*i or j == 2*i+1:
                    continue
                ax.text(j, i, '{:.2f}'.format(arr[i, j]),
                        ha="center", va="center", color="k")
        ax.set_title(roi)
        plt.tight_layout()
        plt.savefig(out_file.format(roi))
    # plt.show()


def plot_prob_map_similarity1():
    """
    样图参见 figures-20210604.pptx Supplemental Figure S3 heatmap
    References:
        1. https://blog.csdn.net/qq_27825451/article/details/105652244
        2. https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        3. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    figsize = (2, 2)
    rois = ('pFus', 'mFus')
    meas = 'corr'  # corr or dice
    data_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                'grouping/prob_map_similarity.pkl')
    gid2name = {
        0: 'single',
        1: 'two-C',
        2: 'two-S'}

    # outputs
    out_file = pjoin(work_dir, 'prob_similarity1_{}.jpg')

    # prepare
    data = pkl.load(open(data_file, 'rb'))
    n_gid = len(data['gid'])
    ticks = np.arange(n_gid)
    gid_names = [gid2name[gid] for gid in data['gid']]

    for roi in rois:
        _, ax = plt.subplots(figsize=figsize)
        k_lh = f'lh_{roi}_{meas}'
        k_rh = f'rh_{roi}_{meas}'
        tril_mask = np.tri(n_gid, k=-1, dtype=bool)
        arr = data[k_rh].copy()
        arr[tril_mask] = data[k_lh][tril_mask]
        diag_mask = np.eye(n_gid, dtype=bool)
        arr[diag_mask] = 0
        ax.imshow(arr, 'autumn')

        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        ax.set_xticks(ticks)
        ax.set_xticklabels(gid_names)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
        ax.set_yticks(ticks)
        ax.set_yticklabels(gid_names)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks(np.arange(n_gid)-.5, minor=True)
        ax.set_yticks(np.arange(n_gid)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(n_gid):
            for j in range(n_gid):
                if i == j:
                    continue
                ax.text(j, i, '{:.2f}'.format(arr[i, j]),
                        ha="center", va="center", color="k")
        # ax.set_title(roi)
        plt.tight_layout()
        plt.savefig(out_file.format(roi))
    # plt.show()


def plot_retest_reliability_icc():
    import numpy as np
    import pickle as pkl
    from nibrain.util.plotfig import auto_bar_width

    # inputs
    figsize = (6.4, 4.8)
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    retest_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                                 'refined_with_Kevin/retest')
    atlas_names = ('MPM', 'ROIv3')
    meas2file = {
        'thickness': pjoin(retest_dir, 'reliability/thickness_{atlas}_rm-subj_icc.pkl'),
        'myelin': pjoin(retest_dir, 'reliability/myelin_{atlas}_rm-subj_icc.pkl'),
        'activ': pjoin(retest_dir, 'reliability/activ_{atlas}_rm-subj_icc.pkl'),
        'rsfc': pjoin(retest_dir, 'rfMRI/{atlas}_rm-subj_icc.pkl'),
    }
    meas2title = {
        'thickness': 'thickness',
        'myelin': 'myelination',
        'activ': 'face selectivity',
        'rsfc': 'RSFC'
    }
    atlas2color = {'MPM': (0.33, 0.33, 0.33, 1),
                   'ROIv3': (0.66, 0.66, 0.66, 1)}

    # outputs
    out_file = pjoin(work_dir, 'retest_reliabilty_icc.jpg')

    # prepare
    n_hemi = len(hemis)
    n_roi = len(rois)
    n_atlas = len(atlas_names)
    n_meas = len(meas2file)
    x = np.arange(n_roi)

    # plot
    _, axes = plt.subplots(n_meas, n_hemi, figsize=figsize)
    offset = -(n_atlas - 1) / 2
    width = auto_bar_width(x, n_atlas)
    for atlas_idx, atlas_name in enumerate(atlas_names):
        for meas_idx, meas_name in enumerate(meas2file.keys()):
            fpath = meas2file[meas_name].format(atlas=atlas_name)
            data = pkl.load(open(fpath, 'rb'))
            for hemi_idx, hemi in enumerate(hemis):
                ax = axes[meas_idx, hemi_idx]
                ys = np.zeros(n_roi)
                yerrs = np.zeros((2, n_roi))
                for roi_idx, roi in enumerate(rois):
                    k = f'{hemi}_{roi}'
                    y = data[k][1]
                    low_err = y - data[k][0]
                    high_err = data[k][2] - y
                    ys[roi_idx] = y
                    yerrs[0, roi_idx] = low_err
                    yerrs[1, roi_idx] = high_err
                ax.bar(x+width*offset, ys, width, yerr=yerrs,
                       label=atlas_name, color=atlas2color[atlas_name])
                if atlas_idx == 1:
                    ax.set_title(meas2title[meas_name])
                    # ax.legend()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xticks(x)
                    ax.set_xticklabels(rois)
                    if hemi_idx == 0:
                        ax.set_ylabel('ICC')
        offset += 1
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def plot_retest_reliability_corr():
    import numpy as np
    import pickle as pkl
    from scipy.stats.stats import sem
    from nibrain.util.plotfig import auto_bar_width

    # inputs
    figsize = (6.4, 4.8)
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    retest_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                                 'refined_with_Kevin/retest')
    atlas_names = ('MPM', 'ROIv3')
    meas2file = {
        'thickness': pjoin(retest_dir, 'reliability/thickness_{atlas}_corr_rm-subj.pkl'),
        'myelin': pjoin(retest_dir, 'reliability/myelin_{atlas}_corr_rm-subj.pkl'),
        'activ': pjoin(retest_dir, 'reliability/activ_{atlas}_corr_rm-subj.pkl'),
        'rsfc': pjoin(retest_dir, 'rfMRI/{atlas}_rm-subj_corr.pkl'),
    }
    meas2title = {
        'thickness': 'thickness',
        'myelin': 'myelination',
        'activ': 'face selectivity',
        'rsfc': 'RSFC'
    }
    atlas2color = {'MPM': (0.33, 0.33, 0.33, 1),
                   'ROIv3': (0.66, 0.66, 0.66, 1)}

    # outputs
    out_file = pjoin(work_dir, 'retest_reliabilty_corr.jpg')

    # prepare
    n_hemi = len(hemis)
    n_roi = len(rois)
    n_atlas = len(atlas_names)
    n_meas = len(meas2file)
    x = np.arange(n_roi)

    # plot
    _, axes = plt.subplots(n_meas, n_hemi, figsize=figsize)
    offset = -(n_atlas - 1) / 2
    width = auto_bar_width(x, n_atlas)
    for atlas_idx, atlas_name in enumerate(atlas_names):
        for meas_idx, meas_name in enumerate(meas2file.keys()):
            fpath = meas2file[meas_name].format(atlas=atlas_name)
            data = pkl.load(open(fpath, 'rb'))
            for hemi_idx, hemi in enumerate(hemis):
                ax = axes[meas_idx, hemi_idx]
                ys = np.zeros(n_roi)
                yerrs = np.zeros(n_roi)
                for roi_idx, roi in enumerate(rois):
                    k = f'{hemi}_{roi}'
                    ys[roi_idx] = np.mean(data[k])
                    yerrs[roi_idx] = sem(data[k])
                ax.bar(x+width*offset, ys, width, yerr=yerrs,
                       label=atlas_name, color=atlas2color[atlas_name])
                if atlas_idx == 1:
                    ax.set_title(meas2title[meas_name])
                    # ax.legend()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xticks(x)
                    ax.set_xticklabels(rois)
                    if hemi_idx == 0:
                        ax.set_ylabel('pearson R')
        offset += 1
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


if __name__ == '__main__':
    # plot_development()
    # plot_development_pattern_corr()
    # plot_prob_map_similarity()
    plot_prob_map_similarity1()
    # plot_retest_reliability_icc()
    # plot_retest_reliability_corr()
