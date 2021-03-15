from os.path import join as pjoin
from matplotlib import pyplot as plt

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
out_dir = pjoin(work_dir, 'paper/Figure3')


def plot_radar():
    """
    https://www.pythoncharts.com/2019/04/16/radar-charts/
    """
    import numpy as np
    import pickle as pkl

    # inputs
    gids = [-1, 1, 2]
    hemis = ('lh', 'rh')
    n_hemi = len(hemis)
    seed_names = ['pFus-face', 'mFus-face']
    seed2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    data_file1 = pjoin(work_dir, 'rfMRI/plot_rsfc_individual2Cole_{hemi}.pkl')
    data_file2 = pjoin(work_dir, 'grouping/rfMRI/'
                                 'plot_rsfc_individual2Cole_G{gid}_{hemi}.pkl')

    # outputs
    out_file = pjoin(out_dir, 'radar_G{}_individual.jpg')

    trg_names = None
    trg_labels = None
    for gid in gids:
        _, axes = plt.subplots(1, n_hemi, subplot_kw=dict(polar=True),
                               num=gid, figsize=(6.4, 4.8))
        for hemi_idx, hemi in enumerate(hemis):
            ax = axes[hemi_idx]
            if gid == -1:
                data = pkl.load(open(data_file1.format(hemi=hemi), 'rb'))
            else:
                data = pkl.load(open(data_file2.format(gid=gid, hemi=hemi),
                                     'rb'))

            if trg_names is None:
                trg_names = data['target']
                trg_labels = data['trg_label']

            indices = [data['target'].index(n) for n in trg_names]
            n_trg = len(trg_names)
            print(n_trg)
            means = data['mean'][:, indices]
            errs = data['sem'][:, indices] * 2

            angles = np.linspace(0, 2 * np.pi, n_trg, endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            means = np.c_[means, means[:, [0]]]
            errs = np.c_[errs, errs[:, [0]]]
            for seed_name in seed_names:
                seed_idx = data['seed'].index(seed_name)
                ax.plot(angles, means[seed_idx], linewidth=1,
                        linestyle='solid', label=seed_name.split('-')[0],
                        color=seed2color[seed_name])
                ax.fill_between(angles, means[seed_idx]-errs[seed_idx],
                                means[seed_idx]+errs[seed_idx],
                                color=seed2color[seed_name], alpha=0.25)
            # ax.legend(loc='upper center')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(trg_labels)
            ax.grid(axis='y')
        plt.tight_layout()
        plt.savefig(out_file.format(gid))

    for lbl, name in zip(trg_labels, trg_names):
        print(lbl, name)


def plot_bar():
    """
    把整组(-1)的和分组(1, 2)的都画到一起
    """
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from nibrain.util.plotfig import auto_bar_width

    # inputs
    gids = (-1, 1, 2)
    hemis = ('lh', 'rh')
    seeds = ('pFus-face', 'mFus-face')
    seed2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    file1 = pjoin(work_dir, 'rfMRI/rsfc_individual2Cole_{hemi}.pkl')
    file2 = pjoin(work_dir, 'grouping/rfMRI/'
                            'rsfc_individual2Cole_G{gid}_{hemi}.pkl')

    # outputs
    out_file = pjoin(out_dir, 'bar_G{}_mean-across-network.jpg')

    n_hemi = len(hemis)
    n_seed = len(seeds)
    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_seed)
    for gid_idx, gid in enumerate(gids):
        plt.figure(gid, figsize=(1.4, 3))
        ax = plt.gca()
        if gid == -1:
            hemi2meas = {
                'lh': pkl.load(open(file1.format(hemi='lh'), 'rb')),
                'rh': pkl.load(open(file1.format(hemi='rh'), 'rb'))}
        else:
            hemi2meas = {
                'lh': pkl.load(open(file2.format(gid=gid, hemi='lh'), 'rb')),
                'rh': pkl.load(open(file2.format(gid=gid, hemi='rh'), 'rb'))}

        offset = -(n_seed - 1) / 2
        for seed in seeds:
            y = np.zeros(n_hemi)
            y_err = np.zeros(n_hemi)
            for hemi_idx, hemi in enumerate(hemis):
                meas = np.mean(hemi2meas[hemi][seed], 1)
                meas = meas[~np.isnan(meas)]
                y[hemi_idx] = np.mean(meas)
                y_err[hemi_idx] = sem(meas)
            ax.bar(x+width*offset, y, width, yerr=y_err,
                   label=seed.split('-')[0], color=seed2color[seed])
            offset += 1
        ax.set_xticks(x)
        ax.set_xticklabels(hemis)
        ax.set_ylabel('RSFC')
        ax.set_ylim(0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        plt.tight_layout()
        plt.savefig(out_file.format(gid))
    # plt.show()


if __name__ == '__main__':
    # plot_radar()
    plot_bar()
