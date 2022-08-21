import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from magicbox.io.io import CiftiReader, save2cifti
from cxy_hcp_ffa.lib.predefine import proj_dir, LR_count_32k,\
    mmp_map_file, mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'NI_R2')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_CNR():
    """
    BOLD CNR (e.g., BOLD Variance divided by Unstructured Noise Variance).
    These two quantities can be determined by regressing out the signal spatial
    ICA component timeseries (from the sICA+FIX processing run by the HCP) from
    the cleaned resting state timeseries (to compute the Unstructured Noise Variance)
    and then taking the difference between the Cleaned Timeseries Variance and
    the Unstructured Noise Variance to compute the BOLD Variance.
    遍历所有1206个被试，对所有状态为ok=(1200, 91282)，并且有对应ICA数据的run做CNR的计算
    为至少有一个有效run的被试存一个.dscalar.nii文件，一共有n_run x 4个map。
    每个run的4个map分别对应这四种指标：'Cleaned Timeseries Variance',
    'Unstructured Noise Variance', 'BOLD Variance', 'BOLD CNR'.
    """
    var_names = ['Cleaned Timeseries Variance', 'Unstructured Noise Variance',
                 'BOLD Variance', 'BOLD CNR']
    runs = ('rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    cleaned_signal_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    melodic_mix_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_hp2000.ica/filtered_func_data.ica/melodic_mix.sdseries.nii'
    out_dir = pjoin(work_dir, 'CNR')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log_file = pjoin(out_dir, 'get_CNR_log')

    n_var = len(var_names)
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'
    n_subj = df_check.shape[0]
    log_writer = open(log_file, 'w')
    for sidx, sid in enumerate(df_check.index, 1):
        time1 = time.time()
        # find valid runs
        run_names = []
        run2cii = {}
        for run in runs:
            if not df_check.loc[sid, run]:
                continue
            melodic_mix_file = melodic_mix_files.format(sid=sid, run=run)
            try:
                cii = nib.load(melodic_mix_file)
                cii.get_data()
            except OSError:
                msg = f'{melodic_mix_file} meets OSError.'
                print(msg)
                log_writer.write(f'{msg}\n')
                continue
            run_names.append(run)
            run2cii[run] = cii

        # go on or not
        n_run = len(run_names)
        if n_run == 0:
            continue
        out_file = pjoin(out_dir, f'{sid}_BOLD-CNR.dscalar.nii')

        # loop runs
        data = None
        bms = None
        vol = None
        map_idx = 0
        map_names = []
        for run_name in run_names:
            # prepare cleaned signal data
            cleaned_signal_file = cleaned_signal_files.format(sid=sid, run=run_name)
            if data is None:
                reader = CiftiReader(cleaned_signal_file)
                bms = reader.brain_models()
                vol = reader.volume
                cleaned_signal = reader.get_data()
                data = np.zeros((n_run * n_var, cleaned_signal.shape[1]))
            else:
                cleaned_signal = nib.load(cleaned_signal_file).get_fdata()

            # prepare signal ICs timeseries
            cii = run2cii[run_name]
            idx_map = cii.header.get_index_map(1)
            signal_indices1 = []
            signal_indices2 = []
            for nm_idx, nm in enumerate(idx_map.named_maps):
                mn = nm.map_name
                if mn.split(' ')[-1] == 'Signal':
                    signal_indices1.append(nm_idx)
                    signal_indices2.append(int(mn.split(':')[0]) - 1)
            assert signal_indices1 == signal_indices2
            ic_timeseries = cii.get_fdata()[:, signal_indices1]
            ic_timeseries = ic_timeseries - np.mean(ic_timeseries, 0, keepdims=True)

            ctv_idx, unv_idx, bv_idx = 0, 0, 0
            for var_name in var_names:
                map_names.append(f'{run_name}_{var_name}')
                if var_name == 'Cleaned Timeseries Variance':
                    data[map_idx] = np.std(cleaned_signal, 0, ddof=1)
                    ctv_idx = map_idx
                elif var_name == 'Unstructured Noise Variance':
                    reg = LinearRegression(fit_intercept=True).fit(
                        ic_timeseries, cleaned_signal)
                    unstrc_noise = cleaned_signal - ic_timeseries.dot(reg.coef_.T)
                    data[map_idx] = np.std(unstrc_noise, 0, ddof=1)
                    unv_idx = map_idx
                elif var_name == 'BOLD Variance':
                    data[map_idx] = data[ctv_idx] - data[unv_idx]
                    bv_idx = map_idx
                elif var_name == 'BOLD CNR':
                    data[map_idx] = data[bv_idx] / data[unv_idx]
                else:
                    raise ValueError
                map_idx += 1
        save2cifti(out_file, data, bms, map_names, vol)
        print(f'Finished {sidx}/{n_subj}, cost {time.time()-time1} seconds.')


def get_CNR_ind_FFA():
    """
    在每个被试中，在各run中计算个体FFA内的平均BOLD CNR，然后计算run间平均。
    最终得到一个pickle文件，键是个体FFA的名字，值是长度为1053的向量。其中NAN
    表示该被试没有对应的FFA。（已经证明这1053个被试都有计算BOLD CNR的有效run）
    """
    roi_names = ['R_pFus', 'R_mFus', 'L_pFus', 'L_mFus']
    roi_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    cnr_files = pjoin(work_dir, 'CNR/{sid}_BOLD-CNR.dscalar.nii')
    out_file = pjoin(work_dir, 'CNR/individual-FFA_BOLD-CNR.pkl')

    reader = CiftiReader(roi_file)
    subj_ids = reader.map_names()
    lbl_tabs = reader.label_tables()
    roi_maps = reader.get_data()

    n_subj, n_vtx = roi_maps.shape
    out_dict = {}
    for roi_name in roi_names:
        out_dict[roi_name] = np.ones(n_subj) * np.nan
    for sidx, sid in enumerate(subj_ids):
        reader = CiftiReader(cnr_files.format(sid=sid))
        cnr_indices = [i for i, j in enumerate(reader.map_names())
                       if j.split('_')[-1] == 'BOLD CNR']
        cnr_maps = reader.get_data()[cnr_indices, :n_vtx]
        for roi_key in lbl_tabs[sidx].keys():
            if roi_key == 0:
                continue
            roi_name = lbl_tabs[sidx][roi_key].label.split('-')[0]
            mask = roi_maps[sidx] == roi_key
            out_dict[roi_name][sidx] = np.mean(np.mean(cnr_maps[:, mask], 1))
        print(f'Finished {sidx + 1}/{n_subj}')

    pkl.dump(out_dict, open(out_file, 'wb'))


def cnr_regression(src_file, cnr_file, out_file):
    """
    将BOLD CNR的个体变异从数据中回归掉
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    rois = ('pFus', 'mFus')

    df = pd.read_csv(src_file)
    n_subj = df.shape[0]
    cnr_data = pkl.load(open(cnr_file, 'rb'))
    out_dict = {}
    for hemi in hemis:
        Hemi = hemi2Hemi[hemi]
        for roi in rois:
            col = f'{hemi}_{roi}'
            meas_vec = np.array(df[col])
            mask = ~np.isnan(meas_vec)
            meas_vec = meas_vec[mask]
            cnr_vec = cnr_data[f'{Hemi}_{roi}'][mask]
            assert np.all(~np.isnan(cnr_vec))
            cnr_vec = cnr_vec - np.mean(cnr_vec)
            cnr_vec = np.expand_dims(cnr_vec, 1)
            reg = LinearRegression(fit_intercept=True).fit(cnr_vec, meas_vec)
            meas_reg = meas_vec - cnr_vec.dot(reg.coef_.T)
            out_dict[col] = np.ones(n_subj) * np.nan
            out_dict[col][mask] = meas_reg

    pd.DataFrame(out_dict).to_csv(out_file, index=False)


def zstat_corr_beta():
    """
    计算每个被试左右FFC内部 zstat pattern和beta pattern的相关
    """
    mask_name = 'FFC'
    roi_names = ('L_FFC', 'R_FFC')
    subj_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    zstat_file = pjoin(anal_dir, 'NI_R1/data_1053/S1200_1053_tfMRI_WM_'
                       'level2_FACE-AVG_hp200_s2_MSMAll.32k_fs_LR.dscalar.nii')
    beta_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/tfMRI_WM/'\
        'tfMRI_WM_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/'\
        'cope20.feat/cope1.dtseries.nii'
    out_dir = pjoin(work_dir, 'zstat_corr_beta')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log_file = pjoin(out_dir, f'corr_in_{mask_name}_log')
    out_file = pjoin(out_dir, f'corr_in_{mask_name}.pkl')

    subj_ids = open(subj_file).read().splitlines()
    # prepare zstat data
    reader = CiftiReader(zstat_file)
    subj_ids_tmp = [i.split('_')[0] for i in reader.map_names()]
    assert subj_ids == subj_ids_tmp
    n_subj = len(subj_ids)
    zstat_maps = reader.get_data()

    # prepare mask and out data
    atlas_map = nib.load(mmp_map_file).get_fdata()[0]
    roi2mask = {}
    out_dict = {}
    for roi in roi_names:
        roi2mask[roi] = atlas_map == mmp_name2label[roi]
        out_dict[roi] = np.ones(n_subj) * np.nan

    log_writer = open(log_file, 'w')
    for subj_idx, subj_id in enumerate(subj_ids):
        zstat_map = zstat_maps[subj_idx]
        beta_file = beta_files.format(sid=subj_id)
        try:
            beta_map = nib.load(beta_file).get_fdata()
        except OSError as err:
            print(err)
            log_writer.write(f'{err}\n')
            continue
        beta_map = beta_map[0, :LR_count_32k]
        for roi in roi_names:
            mask = roi2mask[roi]
            out_dict[roi][subj_idx] = pearsonr(zstat_map[mask], beta_map[mask])[0]
        print(f'Finished {subj_idx+1}/{n_subj}')

    log_writer.close()
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_CNR()
    # get_CNR_ind_FFA()
    # cnr_regression(
    #     src_file=pjoin(anal_dir, 'NI_R1/data_1053/FFA_activ-emo.csv'),
    #     cnr_file=pjoin(work_dir, 'CNR/individual-FFA_BOLD-CNR.pkl'),
    #     out_file=pjoin(work_dir, 'CNR/FFA_activ-emo_clean-CNR.csv'))
    # cnr_regression(
    #     src_file=pjoin(anal_dir, 'NI_R1/data_1053/rsfc_FFA2Cole-mean.csv'),
    #     cnr_file=pjoin(work_dir, 'CNR/individual-FFA_BOLD-CNR.pkl'),
    #     out_file=pjoin(work_dir, 'CNR/rsfc_FFA2Cole-mean_clean-CNR.csv'))
    zstat_corr_beta()
