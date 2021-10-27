from os.path import join as pjoin
from cxy_hcp_ffa.lib.predefine import proj_dir, net2label_cole, \
    mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping/rfMRI')


def pre_ANOVA_3factors():
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 hemispheres x groups x ROIs
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat

    gids = (0, 1, 2)
    hemis = ('lh', 'rh')
    seeds = ('pFus', 'mFus')
    src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    trg_file = pjoin(work_dir, 'rsfc_FFA2Cole_preANOVA-3factor-gid012.csv')

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


def pre_ANOVA_3factors_mix():
    """
    准备好3因素混合设计方差分析需要的数据。
    被试间因子：group
    被试内因子：hemisphere，ROI
    2 groups x 2 hemispheres x 2 ROIs
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat

    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2.csv')
    trg_file = pjoin(work_dir, 'rsfc_FFA2MMP_preANOVA-3factor-mix.csv')

    data = loadmat(src_file)
    gid_df = pd.read_csv(gid_file)
    out_dict = {'gid': []}
    for idx, gid in enumerate(gids):
        gid_idx_vec = np.logical_and(gid_df['lh'] == gid,
                                     gid_df['rh'] == gid)
        nan_vec = None
        for hemi in hemis:
            for roi in rois:
                meas_vec = np.mean(data[f'{hemi}_{roi}'][gid_idx_vec], 1)

                if nan_vec is None:
                    nan_vec = np.isnan(meas_vec)
                    non_nan_vec = ~nan_vec
                    n_valid = np.sum(non_nan_vec)
                    print('#NAN:', np.sum(nan_vec))
                    print(f'G{gid}:', n_valid)
                else:
                    assert np.all(nan_vec == np.isnan(meas_vec))

                meas_vec = meas_vec[non_nan_vec]
                if idx == 0:
                    out_dict[f"{hemi}_{roi.split('-')[0]}"] = meas_vec.tolist()
                else:
                    out_dict[f"{hemi}_{roi.split('-')[0]}"].extend(meas_vec)
        out_dict['gid'].extend([gid] * n_valid)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def roi_ttest(gid, trg_name2label):
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    from scipy.stats.stats import ttest_ind
    from magicbox.stats import EffectSize

    # parameters
    hemis = ('lh', 'rh')
    roi_pair = ('pFus', 'mFus')
    src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    vs_name = f"{roi_pair[0]}_vs_{roi_pair[1]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_FFA2Cole_G{gid}_{vs_name}_ttest.csv")

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

    # save out
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(out_file, index=False)


def roi_pair_ttest(gid, trg_name2label):
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    from scipy.stats.stats import ttest_rel
    from magicbox.stats import EffectSize

    # inputs
    hemis = ('lh', 'rh')
    roi_pair = ('pFus', 'mFus')
    src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat')
    gid_file = pjoin(anal_dir, 'grouping/group_id_v2_merged.csv')
    vs_name = f"{roi_pair[0]}_vs_{roi_pair[1]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_FFA2MMP_G{gid}_{vs_name}_ttest-paired.csv")

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
        for trg_idx, trg_lbl in enumerate(data['target_label'][0]):
            sample1 = data[item1][gid_vec_idx, trg_idx]
            sample2 = data[item2][gid_vec_idx, trg_idx]
            nan_vec1 = np.isnan(sample1)
            nan_vec2 = np.isnan(sample2)
            nan_vec = np.logical_or(nan_vec1, nan_vec2)
            print('#NAN in sample1 or sample2:', np.sum(nan_vec))
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
    data_file = pjoin(work_dir, f'rsfc_FFA2Cole_G{gid}'
                                '_pFus_vs_mFus_ttest.csv')

    # outputs
    out_file = pjoin(work_dir, f'rsfc_FFA2Cole_G{gid}'
                               '_pFus_vs_mFus_ttest_mtc.csv')

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


def mtc_file2cifti():
    import numpy as np
    import pandas as pd
    from magicbox.io.io import CiftiReader, save2cifti
    from cxy_visual_dev.lib.predefine import Atlas, mmp_map_file

    hemis = ('lh', 'rh')
    fpaths = (
        pjoin(work_dir, 'rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest_mtc.csv'),
        pjoin(work_dir, 'rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest_mtc.csv')
    )
    gnames = ('continuous', 'separate')
    out_file = pjoin(work_dir, 'rsfc_FFA2MMP_pFus_vs_mFus_ttest_mtc_cohenD.dscalar.nii')

    atlas = Atlas('HCP_MMP1')
    reader = CiftiReader(mmp_map_file)
    data = np.ones((4, atlas.maps.shape[1]), np.float64) * np.nan
    map_names = []
    row_idx = 0
    for f_idx, fpath in enumerate(fpaths):
        df = pd.read_csv(fpath, index_col='target_name')
        for hemi in hemis:
            es_col = f'CohenD_{hemi}'
            p_col = f'P_{hemi}(fdr_bh)'
            map_names.append(f'{hemi}_{gnames[f_idx]}')
            for idx in df.index:
                if df.loc[idx, p_col] >= 0.05:
                    continue
                roi_idx_map = atlas.maps[0] == atlas.roi2label[idx]
                data[row_idx, roi_idx_map] = df.loc[idx, es_col]
            row_idx += 1

    save2cifti(out_file, data, reader.brain_models(), map_names)


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
    # pre_ANOVA_3factors_mix()
    # roi_ttest(gid=0, trg_name2label=net2label_cole)
    # roi_ttest(gid=1, trg_name2label=net2label_cole)
    # roi_ttest(gid=2, trg_name2label=net2label_cole)
    # roi_pair_ttest(gid=1, trg_name2label=mmp_name2label)
    # roi_pair_ttest(gid=2, trg_name2label=mmp_name2label)
    # multitest_correct_ttest(gid=0)
    # multitest_correct_ttest(gid=1)
    # multitest_correct_ttest(gid=2)
    mtc_file2cifti()

    # old
    # prepare_plot(gid=1, hemi='lh')
    # prepare_plot(gid=1, hemi='rh')
    # prepare_plot(gid=2, hemi='lh')
    # prepare_plot(gid=2, hemi='rh')
