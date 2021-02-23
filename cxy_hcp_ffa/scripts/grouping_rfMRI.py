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


def multitest_correct_ttest(gid=1):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    # inputs
    hemis = ('lh', 'rh')
    data_file = pjoin(work_dir,
                      f'rsfc_individual2Cole_G{gid}_pFus_vs_mFus_ttest.csv')

    # outputs
    out_file = pjoin(work_dir,
                     f'rsfc_individual2Cole_G{gid}_pFus_vs_mFus_ttest_mtc.csv')

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


if __name__ == '__main__':
    # split_rsfc_to_G1G2(hemi='lh')
    # split_rsfc_to_G1G2(hemi='rh')
    roi_ttest(gid=1)
    roi_ttest(gid=2)
    multitest_correct_ttest(gid=1)
    multitest_correct_ttest(gid=2)
