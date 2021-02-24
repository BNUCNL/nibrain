from os.path import join as pjoin
from matplotlib import pyplot as plt

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
trg_dir = pjoin(work_dir, 'paper/Figure2')


def plot():
    """
    把整组(-1)的和分组(1, 2)的都画到一起
    """
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from nibrain.util.plotfig import auto_bar_width

    gids = [-1, 1, 2]
    files1 = [
        pjoin(work_dir, 'structure/individual_thickness_{hemi}.pkl'),
        pjoin(work_dir, 'structure/individual_myelin_{hemi}.pkl'),
        pjoin(work_dir, 'split/tfMRI/activ_{hemi}.pkl')]
    files2 = [
        pjoin(work_dir,
              'grouping/structure/individual_G{gid}_thickness_{hemi}.pkl'),
        pjoin(work_dir,
              'grouping/structure/individual_G{gid}_myelin_{hemi}.pkl'),
        pjoin(work_dir, 'grouping/split/tfMRI/G{gid}_activ_{hemi}.pkl')]
    trg_file = pjoin(trg_dir, 'TMA_diff.svg')
    ylims = [2.7, 1.3, 2]
    ylabels = ['thickness', 'myelination', 'face selectivity']
    hemis = ['lh', 'rh']
    n_hemi = len(hemis)
    rois = ['pFus-face', 'mFus-face']
    roi2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    n_roi = len(rois)
    x = np.arange(n_hemi)
    _, axes = plt.subplots(len(gids), len(ylabels))
    for gid_idx, gid in enumerate(gids):
        files = files1 if gid == -1 else files2
        for f_idx, file in enumerate(files):
            ax = axes[gid_idx, f_idx]
            if gid == -1:
                hemi2meas = {
                    'lh': pkl.load(open(file.format(hemi='lh'), 'rb')),
                    'rh': pkl.load(open(file.format(hemi='rh'), 'rb'))}
            else:
                hemi2meas = {
                    'lh': pkl.load(open(file.format(gid=gid,
                                                    hemi='lh'), 'rb')),
                    'rh': pkl.load(open(file.format(gid=gid, hemi='rh'), 'rb'))
                }

            width = auto_bar_width(x, n_roi)
            offset = -(n_roi - 1) / 2
            for roi in rois:
                y = np.zeros(n_hemi)
                y_err = np.zeros(n_hemi)
                for hemi_idx, hemi in enumerate(hemis):
                    roi_idx = hemi2meas[hemi]['roi'].index(roi)
                    meas = hemi2meas[hemi]['meas'][roi_idx]
                    meas = meas[~np.isnan(meas)]
                    y[hemi_idx] = np.mean(meas)
                    y_err[hemi_idx] = sem(meas)
                ax.bar(x+width*offset, y, width, yerr=y_err,
                       label=roi.split('-')[0], color=roi2color[roi])
                offset += 1
            ax.set_xticks(x)
            ax.set_xticklabels(hemis)
            ax.set_ylabel(ylabels[f_idx])
            ax.set_ylim(ylims[f_idx])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # if 'myelin' in file and gid_idx == 0:
            #     ax.legend()

    plt.tight_layout()
    plt.savefig(trg_file)


if __name__ == '__main__':
    plot()
