import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from scipy.io import loadmat
from scipy.stats.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
from magicbox.stats import EffectSize
from magicbox.io.io import CiftiReader, save2cifti
from cxy_hcp_ffa.lib.predefine import proj_dir, net2label_cole,\
    mmp_name2label, mmp_map_file, LR_count_32k

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')


# ---gdist---
def compare_gdist_grouping(gid_file, gid, items, t_type, data_file):
    """
    比较各组左脑和右脑的FFA间距

    Args:
        gid (int): group ID
        items (_type_): _description_
        t_type (str): the type of t-test
            t: two-sample t-test
            pair-t: paired t-test
        data_file (_type_): _description_
    """
    # prepare group ID information
    gid_df = pd.read_csv(gid_file)
    gid_idx_lh = np.array(gid_df['lh']) == gid
    gid_idx_rh = np.array(gid_df['rh']) == gid

    # calculate
    es = EffectSize()
    fname = os.path.basename(data_file)
    df = pd.read_csv(data_file)
    for item in items:
        print(f'\n---{t_type} between lh and rh {item} in {fname}---')
        col_lh = 'lh_' + item
        col_rh = 'rh_' + item
        if t_type == 't':
            data_lh = np.array(df[col_lh])[gid_idx_lh]
            data_rh = np.array(df[col_rh])[gid_idx_rh]
            print(f"Cohen's d of {col_lh} | {col_rh}:", es.cohen_d(data_lh, data_rh))
            out_info = ttest_ind(data_lh, data_rh)
        elif t_type == 'pair-t':
            gid_idx = np.logical_and(gid_idx_lh, gid_idx_rh)
            data_lh = np.array(df[col_lh])[gid_idx]
            data_rh = np.array(df[col_rh])[gid_idx]
            data = data_lh - data_rh
            print(f"Cohen's d of {col_lh} | {col_rh}:", es.cohen_d(data, [0]))
            out_info = ttest_rel(data_lh, data_rh)
        else:
            raise ValueError
        print(f'average of {col_lh} | {col_rh}: {np.mean(data_lh)} | {np.mean(data_rh)}')
        print(f'size of {col_lh} | {col_rh}: {len(data_lh)} | {len(data_rh)}')
        print(f'{col_lh} vs {col_rh}:', out_info)


def pre_ANOVA_gdist_peak_mix(data_file, gid_file, out_file):
    """
    准备好2因素混合设计方差分析需要的数据。
    被试间因子：group
    被试内因子：hemisphere
    2 groups x 2 hemispheres
    """
    hemis = ('lh', 'rh')
    gids = (1, 2)

    df = pd.read_csv(data_file)
    gid_df = pd.read_csv(gid_file)

    out_dict = {'gid': []}
    for idx, gid in enumerate(gids):
        gid_idx_vec = (gid_df['lh'] == gid) & \
                      (gid_df['rh'] == gid)
        for hemi in hemis:
            col = f'{hemi}_pFus-mFus'
            meas_vec = np.array(df[col][gid_idx_vec])
            assert np.all(~np.isnan(meas_vec))
            if idx == 0:
                out_dict[hemi] = meas_vec.tolist()
            else:
                out_dict[hemi].extend(meas_vec)
        n_valid = np.sum(gid_idx_vec)
        print('#subject of gid:', n_valid)
        out_dict['gid'].extend([gid] * n_valid)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(out_file, index=False)


# ---thickness, myelin, va, face selectivity, rsfc_FFA2Cole-mean---
def pre_ANOVA_3factors_mix(meas_file, gid_file, out_file, gids, rois):
    """
    准备好3因素混合设计方差分析需要的数据。
    被试间因子：group
    被试内因子：hemisphere，ROI
    groups x 2 hemispheres x ROIs

    Args:
        meas_file (str): CSV data
        gid_file (str): CSV group ID
        out_file (str): CSV preANOVA
        gids (sequence): group IDs
        rois (sequence): IOG | pFus | mFus
    """
    hemis = ('lh', 'rh')
    meas_df = pd.read_csv(meas_file)
    gid_df = pd.read_csv(gid_file)

    out_dict = {'gid': []}
    for idx, gid in enumerate(gids):
        gid_idx_vec = (gid_df['lh'] == gid) & \
                      (gid_df['rh'] == gid)
        nan_vec = None
        for hemi in hemis:
            for roi in rois:
                col = f'{hemi}_{roi}'
                meas_vec = np.array(meas_df[col][gid_idx_vec])

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
    out_df.to_csv(out_file, index=False)


# ---rfMRI---
def roi_pair_ttest(src_file, gid_file, out_dir, gid, trg_name2label):
    """
    compare rsfc difference between pFus-faces and mFus-faces
    scheme: hemi-separately network/area-wise
    """
    # inputs
    hemis = ('lh', 'rh')
    roi_pair = ('pFus', 'mFus')
    fname = os.path.basename(src_file).split('.')[0]
    vs_name = f"{roi_pair[0]}_vs_{roi_pair[1]}"

    # outputs
    out_file = pjoin(out_dir, f"{fname}_G{gid}_{vs_name}_ttest-paired.csv")

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
            nan_vec = np.logical_or(nan_vec1, nan_vec2)
            print('#NAN in sample1:', np.sum(nan_vec1))
            print('#NAN in sample2:', np.sum(nan_vec2))
            print('#NAN in sample1 or sample2:', np.sum(nan_vec))
            sample1 = sample1[~nan_vec]
            sample2 = sample2[~nan_vec]
            d = es.cohen_d(sample1 - sample2, [0])
            t, p = ttest_rel(sample1, sample2)
            out_data[f'CohenD_{hemi}'].append(d)
            out_data[f't_{hemi}'].append(t)
            out_data[f'P_{hemi}'].append(p)
            out_data[f'size1_{hemi}'].append(len(sample1))
            out_data[f'size2_{hemi}'].append(len(sample2))

    # save out
    out_data = pd.DataFrame(out_data)
    out_data.to_csv(out_file, index=False)


def multitest_correct_ttest(work_dir, fname):
    # inputs
    hemis = ('lh', 'rh')
    src_file = pjoin(work_dir, fname)
    fname = fname.split('.')[0]
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


def ttest_stats(work_dir, fname):
    hemis = ('lh', 'rh')
    src_file = pjoin(work_dir, fname)
    fname = fname.split('.')[0]
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


def mtc_file2cifti(gnames, fpaths, out_file):

    hemis = ('lh', 'rh')
    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data()[0]
    data = np.ones((4, LR_count_32k), np.float64) * np.nan
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
                roi_idx_map = mmp_map == mmp_name2label[idx]
                data[row_idx, roi_idx_map] = df.loc[idx, es_col]
            row_idx += 1

    save2cifti(out_file, data, reader.brain_models(), map_names)


if __name__ == '__main__':
    # compare_gdist_grouping(
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     gid=2, items=('pFus-mFus',), t_type='t',
    #     data_file=pjoin(anal_dir, 'gdist_min1.csv'))
    # compare_gdist_grouping(
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     gid=2, items=('pFus-mFus',), t_type='pair-t',
    #     data_file=pjoin(anal_dir, 'gdist_min1.csv'))
    # compare_gdist_grouping(
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     gid=2, items=('pFus-mFus',), t_type='t',
    #     data_file=pjoin(anal_dir, 'NI_R1/data_1053/gdist_min1.csv'))
    # compare_gdist_grouping(
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     gid=2, items=('pFus-mFus',), t_type='pair-t',
    #     data_file=pjoin(anal_dir, 'NI_R1/data_1053/gdist_min1.csv'))

    # pre_ANOVA_gdist_peak_mix(
    #     data_file=pjoin(anal_dir, 'gdist_peak.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'gdist_peak_preANOVA_mix.csv')
    # )
    # pre_ANOVA_gdist_peak_mix(
    #     data_file=pjoin(anal_dir, 'NI_R1/data_1053/gdist_peak.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/gdist_peak_preANOVA_mix.csv')
    # )

    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_thickness.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/structure/FFA_thickness_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_myelin.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/structure/FFA_myelin_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_va.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/structure/FFA_va_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole-mean.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/rfMRI/rsfc_FFA2Cole-mean_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_thickness.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_thickness_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_myelin.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_myelin_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_va.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_va_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole-mean.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole-mean_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole-mean_clean-TSNR2.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole-mean_clean-TSNR2_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    pre_ANOVA_3factors_mix(
        meas_file=pjoin(anal_dir, 'NI_R2/CNR/rsfc_FFA2Cole-mean_clean-CNR.csv'),
        gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
        out_file=pjoin(anal_dir, 'NI_R2/CNR/rsfc_FFA2Cole-mean_clean-CNR_preANOVA-3factor-mix.csv'),
        gids=(1, 2), rois=('pFus', 'mFus')
    )

    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'tfMRI/FFA_activ.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/tfMRI/FFA_activ_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'tfMRI/FFA_activ-emo.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'grouping/tfMRI/FFA_activ-emo_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_activ-emo.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_activ-emo_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors_mix(
    #     meas_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_activ-emo_clean-TSNR2.csv'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_activ-emo_clean-TSNR2_preANOVA-3factor-mix.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    pre_ANOVA_3factors_mix(
        meas_file=pjoin(anal_dir, 'NI_R2/CNR/FFA_activ-emo_clean-CNR.csv'),
        gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
        out_file=pjoin(anal_dir, 'NI_R2/CNR/FFA_activ-emo_clean-CNR_preANOVA-3factor-mix.csv'),
        gids=(1, 2), rois=('pFus', 'mFus')
    )

    # roi_pair_ttest(
    #     src_file=pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     gid=1, trg_name2label=mmp_name2label)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     gid=2, trg_name2label=mmp_name2label)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     gid=1, trg_name2label=net2label_cole)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     gid=2, trg_name2label=net2label_cole)
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest-paired.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'grouping/rfMRI'),
    #     fname='rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest-paired_mtc.csv')

    # roi_pair_ttest(
    #     src_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2MMP.mat'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     gid=1, trg_name2label=mmp_name2label)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2MMP.mat'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     gid=2, trg_name2label=mmp_name2label)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole.mat'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     gid=1, trg_name2label=net2label_cole)
    # roi_pair_ttest(
    #     src_file = pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole.mat'),
    #     gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
    #     out_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     gid=2, trg_name2label=net2label_cole)
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest-paired.csv')
    # multitest_correct_ttest(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest-paired.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2Cole_G1_pFus_vs_mFus_ttest-paired_mtc.csv')
    # ttest_stats(
    #     work_dir=pjoin(anal_dir, 'NI_R1/data_1053'),
    #     fname='rsfc_FFA2Cole_G2_pFus_vs_mFus_ttest-paired_mtc.csv')

    # mtc_file2cifti(
    #     gnames = ('continuous', 'separate'),
    #     fpaths = (
    #         pjoin(anal_dir, 'grouping/rfMRI/rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest_mtc.csv'),
    #         pjoin(anal_dir, 'grouping/rfMRI/rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest_mtc.csv')),
    #     out_file = pjoin(anal_dir, 'grouping/rfMRI/rsfc_FFA2MMP_pFus_vs_mFus_ttest_mtc_cohenD.dscalar.nii')
    # )
    # mtc_file2cifti(
    #     gnames = ('continuous', 'separate'),
    #     fpaths = (
    #         pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2MMP_G1_pFus_vs_mFus_ttest-paired_mtc.csv'),
    #         pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2MMP_G2_pFus_vs_mFus_ttest-paired_mtc.csv')),
    #     out_file = pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2MMP_pFus_vs_mFus_ttest-paired_mtc_cohenD.dscalar.nii')
    # )
