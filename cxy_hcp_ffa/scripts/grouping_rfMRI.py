import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from scipy.io import loadmat
from scipy.stats.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from magicbox.stats import EffectSize
from cxy_hcp_ffa.lib.predefine import proj_dir, net2label_cole, \
    mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping/rfMRI')

def pre_ANOVA_3factors():
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 hemispheres x groups x ROIs
    """
    gids = (0, 1, 2)
    hemis = ('lh', 'rh')
    seeds = ('pFus', 'mFus')
    src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    trg_file = pjoin(work_dir, 'rsfc_FFA2Cole_preANOVA-3factor-gid012_new.csv')

    data = loadmat(src_file)
    gid_df = pd.read_csv(gid_file)
    out_dict = {'hemi': [], 'gid': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        gid_vec = np.array(gid_df[hemi])
        for gid in gids:
            gid_vec_idx = gid_vec == gid
            for seed in seeds:
                col = f'{hemi}_{seed}'
                meas_vec = np.mean(data[col][gid_vec_idx], axis=1)
                meas_vec = meas_vec[~np.isnan(meas_vec)]
                n_valid = len(meas_vec)
                out_dict['hemi'].extend([hemi] * n_valid)
                out_dict['gid'].extend([gid] * n_valid)
                out_dict['roi'].extend([seed] * n_valid)
                out_dict['meas'].extend(meas_vec)
                print(f'{hemi}_{gid}_{seed}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def roi_ttest(src_file, gid, trg_name2label):
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network/area-wise
    """
    # parameters
    hemis = ('lh', 'rh')
    roi_pair = ('pFus', 'mFus')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    fname = os.path.basename(src_file).split('.')[0]
    vs_name = f"{roi_pair[0]}_vs_{roi_pair[1]}"

    # outputs
    out_file = pjoin(work_dir, f"{fname}_G{gid}_{vs_name}_ttest.csv")

    # start
    data = loadmat(src_file)
    gid_df = pd.read_csv(gid_file)
    trg_label2name = {}
    for k, v in trg_name2label.items():
        trg_label2name[v] = k
    out_data = {}
    out_data['target_name'] = [trg_label2name[lbl]
                               for lbl in data['target_label'][0]]
    es = EffectSize()
    for hemi in hemis:
        gid_vec_idx = np.array(gid_df[hemi]) == gid
        item1 = f'{hemi}_{roi_pair[0]}'
        item2 = f'{hemi}_{roi_pair[1]}'
        out_data[f'CohenD_{hemi}'] = []
        out_data[f't_{hemi}'] = []
        out_data[f'P_{hemi}'] = []
        out_data[f'size1_{hemi}'] = []
        out_data[f'size2_{hemi}'] = []
        for trg_idx, trg_lbl in enumerate(data['target_label'][0]):
            sample1 = data[item1][gid_vec_idx, trg_idx]
            sample2 = data[item2][gid_vec_idx, trg_idx]
            nan_vec1 = np.isnan(sample1)
            nan_vec2 = np.isnan(sample2)
            print('#NAN in sample1:', np.sum(nan_vec1))
            print('#NAN in sample2:', np.sum(nan_vec2))
            sample1 = sample1[~nan_vec1]
            sample2 = sample2[~nan_vec2]
            d = es.cohen_d(sample1, sample2)
            t, p = ttest_ind(sample1, sample2)
            out_data[f'CohenD_{hemi}'].append(d)
            out_data[f't_{hemi}'].append(t)
            out_data[f'P_{hemi}'].append(p)
            out_data[f'size1_{hemi}'].append(len(sample1))
            out_data[f'size2_{hemi}'].append(len(sample2))

    # save out
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(out_file, index=False)


def multitest_correct_ttest(src_file):
    # inputs
    hemis = ('lh', 'rh')
    fname = os.path.basename(src_file).split('.')[0]
    out_file = pjoin(work_dir, f'{fname}_mtc.csv')

    # start
    data = pd.read_csv(src_file)
    for hemi in hemis:
        item = f'P_{hemi}'
        ps = np.asarray(data[item])
        reject, ps_fdr, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'fdr_bh')
        reject, ps_bonf, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'bonferroni')
        data[f'{item}(fdr_bh)'] = ps_fdr
        data[f'{item}(bonf)'] = ps_bonf

    # save out
    data.to_csv(out_file, index=False)


def ttest_stats(src_file):
    hemis = ('lh', 'rh')
    fname = os.path.basename(src_file).split('.')[0]
    out_file = pjoin(work_dir, f'{fname}_stats.txt')

    df = pd.read_csv(src_file)
    wf = open(out_file, 'w')

    wf.write(f'#targets: {df.shape[0]}\n')
    for hemi in hemis:
        wf.write(f'\n==={hemi}===\n')

        size1s = np.unique(df[f'size1_{hemi}'])
        assert len(size1s) == 1
        wf.write(f'size1: {size1s[0]}\t')

        size2s = np.unique(df[f'size2_{hemi}'])
        assert len(size2s) == 1
        wf.write(f'size2: {size2s[0]}\n')

        sig_vec = df[f'P_{hemi}(fdr_bh)'] < 0.05
        non_sig_vec = ~sig_vec

        wf.write(f'\n---P(fdr_bh) < 0.05---\n')
        n_sig = np.sum(sig_vec)
        wf.write(f'#targets: {n_sig}\n')
        if n_sig != 0:
            sig_df = df.loc[sig_vec, :]
            pos_vec = sig_df[f't_{hemi}'] > 0
            neg_vec = ~pos_vec

            wf.write('\n***ROI1 > ROI2***\n')
            n_pos = np.sum(pos_vec)
            wf.write(f'#targets: {n_pos}\n')
            if n_pos != 0:
                pos_df = sig_df.loc[pos_vec, :]
                ds = pos_df[f'CohenD_{hemi}']
                ts = pos_df[f't_{hemi}']
                ps = pos_df[f'P_{hemi}(fdr_bh)']
                wf.write(f"CohenD (min): {np.min(ds)}\n")
                wf.write(f"t (min): {np.min(ts)}\n")
                wf.write(f"P(fdr_bh) (max): {np.max(ps)}\n")
                wf.write(f"target name: {' | '.join(pos_df['target_name'])}\n")
                wf.write(f"CohenD: {' | '.join([str(i) for i in ds])}\n")
                wf.write(f"t: {' | '.join([str(i) for i in ts])}\n")
                wf.write(f"P(fdr_bh): {' | '.join([str(i) for i in ps])}\n")
            
            wf.write('\n***ROI1 < ROI2***\n')
            n_neg = np.sum(neg_vec)
            wf.write(f'#targets: {n_neg}\n')
            if n_neg != 0:
                neg_df = sig_df.loc[neg_vec, :]
                ds = neg_df[f'CohenD_{hemi}']
                ts = neg_df[f't_{hemi}']
                ps = neg_df[f'P_{hemi}(fdr_bh)']
                wf.write(f"CohenD (max): {np.max(ds)}\n")
                wf.write(f"t (max): {np.max(ts)}\n")
                wf.write(f"P(fdr_bh) (max): {np.max(ps)}\n")
                wf.write(f"target name: {' | '.join(neg_df['target_name'])}\n")
                wf.write(f"CohenD: {' | '.join([str(i) for i in ds])}\n")
                wf.write(f"t: {' | '.join([str(i) for i in ts])}\n")
                wf.write(f"P(fdr_bh): {' | '.join([str(i) for i in ps])}\n")

        wf.write(f'\n---P(fdr_bh) >= 0.05---\n')
        n_not_sig = np.sum(non_sig_vec)
        wf.write(f'#targets: {n_not_sig}\n')
        if n_not_sig != 0:
            not_sig_df = df.loc[non_sig_vec, :]
            ds = not_sig_df[f'CohenD_{hemi}']
            ts = not_sig_df[f't_{hemi}']
            ps = not_sig_df[f'P_{hemi}(fdr_bh)']
            wf.write(f"CohenD (min, max): {np.min(ds)}, {np.max(ds)}\n")
            wf.write(f"t (min, max): {np.min(ts)}, {np.max(ts)}\n")
            wf.write(f"P(fdr_bh) (min): {np.min(ps)}\n")
            wf.write(f"target name: {' | '.join(not_sig_df['target_name'])}\n")
            wf.write(f"CohenD: {' | '.join([str(i) for i in ds])}\n")
            wf.write(f"t: {' | '.join([str(i) for i in ts])}\n")
            wf.write(f"P(fdr_bh): {' | '.join([str(i) for i in ps])}\n")
    wf.close()


# ---old---
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
    # pre_ANOVA_3factors()
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat'),
    #     gid=0, trg_name2label=net2label_cole)
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat'),
    #     gid=1, trg_name2label=net2label_cole)
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat'),
    #     gid=2, trg_name2label=net2label_cole)
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
    #     gid=0, trg_name2label=mmp_name2label)
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
    #     gid=1, trg_name2label=mmp_name2label)
    # roi_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
    #     gid=2, trg_name2label=mmp_name2label)
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2Cole_G0_pFus_vs_mFus_ttest.csv'))
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest.csv'))
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest.csv'))
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2MMP_G0_pFus_vs_mFus_ttest.csv'))
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest.csv'))
    # multitest_correct_ttest(pjoin(work_dir, 'rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2Cole_G0_pFus_vs_mFus_ttest_mtc.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest_mtc.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest_mtc.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2MMP_G0_pFus_vs_mFus_ttest_mtc.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest_mtc.csv'))
    ttest_stats(pjoin(work_dir, 'rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest_mtc.csv'))

    # old
    # prepare_plot(gid=1, hemi='lh')
    # prepare_plot(gid=1, hemi='rh')
    # prepare_plot(gid=2, hemi='lh')
    # prepare_plot(gid=2, hemi='rh')
