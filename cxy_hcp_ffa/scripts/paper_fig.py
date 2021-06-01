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
            if meas_idx+1 == n_meas:
                ax.set_xlabel(age_name)
            ax.set_ylabel(meas2ylabel[meas_name])
            x_ticks = age_uniq[::2]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


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
    figsize = (4, 4)
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
            arr = np.insert(arr, i*2, data[k_rh][:, i], axis=1)
        img = ax.imshow(arr, 'autumn', aspect=aspect)

        ax.set_xticks(ticks * 2 + 0.5)
        ax.set_xticklabels(gid_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_yticks(ticks)
        ax.set_yticklabels(gid_names)

        for i in range(n_gid):
            for j in range(n_gid*2):
                ax.text(j, i, '{:.2f}'.format(arr[i, j]),
                        ha="center", va="center", color="k")
        ax.set_title(roi)
        plt.savefig(out_file.format(roi))
    # plt.show()


if __name__ == '__main__':
    # plot_development()
    plot_prob_map_similarity()
