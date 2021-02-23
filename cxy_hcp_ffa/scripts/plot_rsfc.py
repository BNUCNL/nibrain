from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
out_dir = pjoin(work_dir, 'paper/Figure3')


def plot_radar():
    """
    https://www.pythoncharts.com/2019/04/16/radar-charts/
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

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


if __name__ == '__main__':
    plot_radar()
