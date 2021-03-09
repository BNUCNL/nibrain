from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/grouping/rfMRI')


def split_rsfc_to_G1G2(hemi='lh'):
    """
    split all individual RSFC to G1 and G2
    """
    import numpy as np
    import pickle as pkl

    # inputs
    src_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rfMRI/rsfc_individual2Cole_{hemi}.pkl')
    gids = (1, 2)
    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'grouping/group_id_{hemi}.npy')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')

    # outputs
    out_file = pjoin(work_dir, 'rsfc_individual2Cole_G{gid}_{hemi}.pkl')

    gid_vec = np.load(gid_file)
    data = pkl.load(open(src_file, 'rb'))
    for gid in gids:
        gid_indices = np.where(gid_vec == gid)[0]
        out_dict = {}
        for k, v in data.items():
            if k == 'subject':
                out_dict[k] = [v[i] for i in gid_indices]
                print(f'{hemi}_{gid}_{k}:', len(out_dict[k]))
            elif k in rois:
                out_dict[k] = v[gid_indices]
                print(f'{hemi}_{gid}_{k}:', out_dict[k].shape)
            else:
                out_dict[k] = v
        pkl.dump(out_dict, open(out_file.format(gid=gid, hemi=hemi), 'wb'))


def pre_ANOVA_3factors():
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 groups x 2 hemispheres x 2 ROIs
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'rsfc_individual2Cole_G{}_{}.pkl')
    trg_file = pjoin(work_dir, 'rsfc_individual2Cole_preANOVA-3factor.csv')

    out_dict = {'gid': [], 'hemi': [], 'roi': [], 'meas': []}
    for gid in gids:
        for hemi in hemis:
            data = pkl.load(open(src_file.format(gid, hemi), 'rb'))
            for roi in rois:
                meas_vec = np.mean(data[roi], axis=1)
                meas_vec = meas_vec[~np.isnan(meas_vec)]
                n_valid = len(meas_vec)
                out_dict['gid'].extend([gid] * n_valid)
                out_dict['hemi'].extend([hemi] * n_valid)
                out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
                out_dict['meas'].extend(meas_vec)
                print(f'{gid}_{hemi}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def roi_ttest(gid=1):
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pickle as pkl
    import pandas as pd
    from scipy.stats.stats import ttest_ind
    from cxy_hcp_ffa.lib.predefine import net2label_cole
    from commontool.stats import EffectSize

    # parameters
    hemis = ('lh', 'rh')
    roi_pair = ('pFus-face', 'mFus-face')
    data_file = pjoin(work_dir, 'rsfc_individual2Cole_G{gid}_{hemi}.pkl')
    vs_name = f"{roi_pair[0].split('-')[0]}_vs_{roi_pair[1].split('-')[0]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_individual2Cole_G{gid}_{vs_name}_ttest.csv")

    # start
    trg_names = list(net2label_cole.keys())
    trg_labels = list(net2label_cole.values())
    out_data = {'network': trg_names}
    es = EffectSize()
    for hemi in hemis:
        data = pkl.load(open(data_file.format(gid=gid, hemi=hemi), 'rb'))
        assert data['trg_label'] == trg_labels

        out_data[f'CohenD_{hemi}'] = []
        out_data[f't_{hemi}'] = []
        out_data[f'P_{hemi}'] = []
        for trg_idx, trg_name in enumerate(trg_names):
            sample1 = data[roi_pair[0]][:, trg_idx]
            sample2 = data[roi_pair[1]][:, trg_idx]
            nan_vec1 = np.isnan(sample1)
            nan_vec2 = np.isnan(sample2)
            print(f'#NAN in sample1:', np.sum(nan_vec1))
            print(f'#NAN in sample2:', np.sum(nan_vec2))
            sample1 = sample1[~nan_vec1]
            sample2 = sample2[~nan_vec2]
            d = es.cohen_d(sample1, sample2)
            t, p = ttest_ind(sample1, sample2)
            out_data[f'CohenD_{hemi}'].append(d)
            out_data[f't_{hemi}'].append(t)
            out_data[f'P_{hemi}'].append(p)

    # save out
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(out_file, index=False)


def roi_pair_ttest(gid=1):
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pickle as pkl
    import pandas as pd
    from scipy.stats.stats import ttest_rel
    from cxy_hcp_ffa.lib.predefine import net2label_cole
    from commontool.stats import EffectSize

    # inputs
    hemis = ('lh', 'rh')
    roi_pair = ('pFus-face', 'mFus-face')
    data_file = pjoin(work_dir, 'rsfc_individual2Cole_G{}_{}.pkl')
    vs_name = f"{roi_pair[0].split('-')[0]}_vs_{roi_pair[1].split('-')[0]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_individual2Cole_G{gid}_{vs_name}_ttest_paired.csv")

    # start
    trg_names = list(net2label_cole.keys())
    trg_labels = list(net2label_cole.values())
    out_data = {'network': trg_names}
    es = EffectSize()
    for hemi in hemis:
        data = pkl.load(open(data_file.format(gid, hemi), 'rb'))
        assert data['trg_label'] == trg_labels

        out_data[f'CohenD_{hemi}'] = []
        out_data[f't_{hemi}'] = []
        out_data[f'P_{hemi}'] = []
        for trg_idx, trg_name in enumerate(trg_names):
            sample1 = data[roi_pair[0]][:, trg_idx]
            sample2 = data[roi_pair[1]][:, trg_idx]
            nan_vec1 = np.isnan(sample1)
            nan_vec2 = np.isnan(sample2)
            nan_vec = np.logical_or(nan_vec1, nan_vec2)
            print(f'#NAN in sample1 or sample2:', np.sum(nan_vec))
            sample1 = sample1[~nan_vec]
            sample2 = sample2[~nan_vec]
            d = es.cohen_d(sample1, sample2)
            t, p = ttest_rel(sample1, sample2)
            out_data[f'CohenD_{hemi}'].append(d)
            out_data[f't_{hemi}'].append(t)
            out_data[f'P_{hemi}'].append(p)

    # save out
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(out_file, index=False)


def multitest_correct_ttest(gid=1):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    # inputs
    hemis = ('lh', 'rh')
    data_file = pjoin(work_dir, f'rsfc_individual2Cole_G{gid}'
                                '_pFus_vs_mFus_ttest_paired.csv')

    # outputs
    out_file = pjoin(work_dir, f'rsfc_individual2Cole_G{gid}'
                               '_pFus_vs_mFus_ttest_paired_mtc.csv')

    # start
    data = pd.read_csv(data_file)
    for hemi in hemis:
        item = f'P_{hemi}'
        ps = np.asarray(data[item])
        reject, ps_fdr, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'fdr_bh')
        reject, ps_bonf, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'bonferroni')
        data[f'{item}(fdr_bh)'] = ps_fdr
        data[f'{item}(bonf)'] = ps_bonf

    # save out
    data.to_csv(out_file, index=False)


def prepare_plot(gid=1, hemi='lh'):
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from cxy_hcp_ffa.lib.predefine import net2label_cole

    # inputs
    data_file = pjoin(work_dir, f'rsfc_individual2Cole_G{gid}_{hemi}.pkl')

    # outputs
    out_file = pjoin(work_dir, f'plot_rsfc_individual2Cole_G{gid}_{hemi}.pkl')

    # load data
    data = pkl.load(open(data_file, 'rb'))
    trg_names = list(net2label_cole.keys())
    trg_labels = list(net2label_cole.values())
    assert data['trg_label'] == trg_labels

    # prepare seed_names and trg_names
    seed_names = ['IOG-face', 'pFus-face', 'mFus-face']
    n_seed = len(seed_names)
    n_trg = len(trg_names)

    # calculate mean and sem
    means = np.ones((n_seed, n_trg)) * np.nan
    sems = np.ones((n_seed, n_trg)) * np.nan
    stds = np.ones((n_seed, n_trg)) * np.nan
    for seed_idx, seed_name in enumerate(seed_names):
        for trg_idx in range(n_trg):
            samples = data[seed_name][:, trg_idx]
            samples = samples[~np.isnan(samples)]
            means[seed_idx, trg_idx] = np.mean(samples)
            sems[seed_idx, trg_idx] = sem(samples)
            stds[seed_idx, trg_idx] = np.std(samples, ddof=1)

    out_dict = {
        'shape': 'n_seed x n_target',
        'seed': seed_names,
        'target': trg_names,
        'trg_label': trg_labels,
        'mean': means,
        'sem': sems,
        'std': stds
    }
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # split_rsfc_to_G1G2(hemi='lh')
    # split_rsfc_to_G1G2(hemi='rh')
    # roi_ttest(gid=1)
    # roi_ttest(gid=2)
    # multitest_correct_ttest(gid=1)
    # multitest_correct_ttest(gid=2)
    # prepare_plot(gid=1, hemi='lh')
    # prepare_plot(gid=1, hemi='rh')
    # prepare_plot(gid=2, hemi='lh')
    # prepare_plot(gid=2, hemi='rh')
    # roi_pair_ttest(gid=1)
    # roi_pair_ttest(gid=2)
    # multitest_correct_ttest(gid=1)
    # multitest_correct_ttest(gid=2)
    pre_ANOVA_3factors()
