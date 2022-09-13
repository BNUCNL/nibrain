import os
import time
import gdist
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.io import loadmat
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from magicbox.io.io import CiftiReader, save2cifti, GiftiReader,\
    save2nifti
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor,\
    label_edge_detection
from cxy_hcp_ffa.lib.predefine import proj_dir, LR_count_32k,\
    mmp_map_file, mmp_name2label, hemi2stru, MedialWall,\
    s1200_midthickness_L, s1200_midthickness_R, mmp_label2name

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


def get_non_FFA_prob():
    """
    基于separate组做出FFC中Non_FFA的概率图
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    ffa_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    gid_file = pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv')
    out_file = pjoin(work_dir, 'nonFFA-in-FFC_prob.dscalar.nii')

    df = pd.read_csv(gid_file)
    reader = CiftiReader(ffa_file)
    mmp_reader = CiftiReader(mmp_map_file)
    data = []
    for hemi in hemis:
        g2_idx_vec = df[hemi] == 2
        ffa_maps, _, _ = reader.get_data(hemi2stru[hemi])
        prob_map = np.mean(ffa_maps[g2_idx_vec] == 0, 0, keepdims=True)
        mmp_map, _, _ = mmp_reader.get_data(hemi2stru[hemi])
        ffc_mask = mmp_map == mmp_name2label[f'{hemi2Hemi[hemi]}_FFC']
        prob_map[~ffc_mask] = np.nan
        data.append(prob_map)

    data = np.concatenate(data, axis=1)
    save2cifti(out_file, data, reader.brain_models())


def get_grp_FFA_gap(thr):
    """
    给基于separate组做出的FFC中Non_FFA的概率图加个阈限(thr)得到gap ROI
    """
    hemis = ('lh', 'rh')
    prob_file = pjoin(work_dir, 'nonFFA-in-FFC_prob.dscalar.nii')
    out_files = pjoin(work_dir, 'FFA-gap_thr-{thr}_{hemi}.nii.gz')

    reader = CiftiReader(prob_file)
    for hemi in hemis:
        prob_map = reader.get_data(hemi2stru[hemi], True)[0]
        data = (prob_map > thr).astype(np.uint8)
        save2nifti(out_files.format(thr=thr, hemi=hemi),
                   np.expand_dims(data, (1, 2)))


def get_FFA_gap():
    """
    在各个被试上，用组gap area减去pFus和mFus的点，剩下的就是个体特异的gap area。
    结果保存在.dlabel.nii文件中：
    1. 包括所有至少有一个半脑属于separate组的被试
    2. 不属于separate组的半脑的数值置为0；在属于separate组的半脑中计算gap area
    3. 某个被试的左/右半脑属于separate组，并且gap area顶点数量不为0，则对应map name带上l/r的标记
    """
    gap_files = pjoin(work_dir, 'FFA-gap_{hemi}.nii.gz')
    out_file = pjoin(work_dir, 'FFA+gap_indiv.32k_fs_LR.dlabel.nii')

    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    ffa_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    gid_file = pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv')

    roi2key = {
        '???': 0,
        'R_pFus-faces': 1,
        'R_mFus-faces': 2,
        'L_pFus-faces': 3,
        'L_mFus-faces': 4,
        'R_FFA-gap': 5,
        'L_FFA-gap': 6}
    gap_rgba = [1., 0., 0., 1.]

    df = pd.read_csv(gid_file)
    g2_idx_vec = np.logical_or(df['lh'] == 2, df['rh'] == 2)
    df = df.loc[g2_idx_vec].reset_index(drop=True)
    reader = CiftiReader(ffa_file)
    bms = reader.brain_models()
    mns = [j for i, j in enumerate(reader.map_names()) if g2_idx_vec[i]]
    lbl_tabs = [j for i, j in enumerate(reader.label_tables()) if g2_idx_vec[i]]
    data = []
    for hemi in hemis:
        data_hemi, _, idx2vtx = reader.get_data(hemi2stru[hemi])
        data_hemi = data_hemi[g2_idx_vec]

        gap_mask_grp = nib.load(
            gap_files.format(hemi=hemi)).get_fdata().squeeze()[idx2vtx]
        gap_name = f'{hemi2Hemi[hemi]}_FFA-gap'
        gap_key = roi2key[gap_name]
        for idx, lbl_tab in enumerate(lbl_tabs):
            if df.loc[idx, hemi] == 2:
                gap_mask = np.logical_and(data_hemi[idx] == 0, gap_mask_grp)
                if not np.any(gap_mask):
                    continue
                data_hemi[idx][gap_mask] = gap_key
                lbl_tab[gap_key] = \
                    nib.cifti2.Cifti2Label(gap_key, gap_name, *gap_rgba)
                mns[idx] = mns[idx] + hemi[0]
            else:
                data_hemi[idx] = 0
                invalid_keys = []
                for k, v in lbl_tab.items():
                    assert roi2key[v.label] == k
                    if v.label.startswith(f'{hemi2Hemi[hemi]}_'):
                        invalid_keys.append(k)
                for k in invalid_keys:
                    lbl_tab.pop(k)
        data.append(data_hemi)

    data = np.concatenate(data, axis=1)
    save2cifti(out_file, data, bms, mns, label_tables=lbl_tabs)


def get_FFA_gap1():
    """
    在各个separate组半脑上，以一定规则扩张pFus和mFus，
    取两者扩张部分的交集作为它们之间的gap area。扩张过程以pFus为例：
    1. 以pFus的内轮廓为扩张的起点
    2. 遍历各扩张起点，对于每个起点，合并1环近邻中距离mFus比它更近的，
        并且既不属于pFus，也不属于mFus的点，同时作为下一步扩张的起点。
    3. 重复第2步，直到没有扩张起点为止。

    结果保存在.dlabel.nii文件中：
    1. 包括所有至少有一个半脑属于separate组的被试
    2. 不属于separate组的半脑的数值置为0；在属于separate组的半脑中计算gap area
    3. 某个被试的左/右半脑属于separate组，并且gap area顶点数量不为0，
        则对应map name带上l/r的标记。
    """
    # settings
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    hemi2gii = {'lh': s1200_midthickness_L, 'rh': s1200_midthickness_R}
    ffa_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    gid_file = pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv')
    out_file = pjoin(work_dir, 'FFA+gap1_indiv.32k_fs_LR.dlabel.nii')
    log_file = pjoin(work_dir, 'FFA+gap1_indiv_log')

    roi2key = {
        '???': 0,
        'R_pFus-faces': 1,
        'R_mFus-faces': 2,
        'L_pFus-faces': 3,
        'L_mFus-faces': 4,
        'R_FFA-gap': 5,
        'L_FFA-gap': 6}
    gap_rgba = [1., 0., 0., 1.]

    # loading
    df = pd.read_csv(gid_file)
    # g2_idx_vec = np.zeros(df.shape[0], dtype=bool)
    # g2_idx_vec[:2] = True
    g2_idx_vec = np.logical_or(df['lh'] == 2, df['rh'] == 2)
    df = df.loc[g2_idx_vec].reset_index(drop=True)
    n_subj = df.shape[0]
    reader = CiftiReader(ffa_file)
    bms = reader.brain_models()
    mns = [j for i, j in enumerate(reader.map_names()) if g2_idx_vec[i]]
    lbl_tabs = [j for i, j in enumerate(reader.label_tables()) if g2_idx_vec[i]]
    data = []
    mw = MedialWall()
    log_writer = open(log_file, 'w')

    # calculating
    for hemi in hemis:
        # prepare individual FFAs
        data_hemi = reader.get_data(hemi2stru[hemi], True)
        _, _, idx2vtx = reader.get_data(hemi2stru[hemi])
        data_hemi = data_hemi[g2_idx_vec]

        # prepare geometry infomation
        gii = GiftiReader(hemi2gii[hemi])
        coords = gii.coords.astype(np.float64)
        faces = gii.faces.astype(np.int32)
        faces = mw.remove_from_faces(hemi, faces)
        vertices = np.arange(data_hemi.shape[1], dtype=np.int32)
        neighbors_list = get_n_ring_neighbor(faces, 1)

        # prepare FFA and gap names
        pfus_name = f'{hemi2Hemi[hemi]}_pFus-faces'
        mfus_name = f'{hemi2Hemi[hemi]}_mFus-faces'
        ffa_pairs = [(pfus_name, mfus_name),  # expand pFus to mFus
                     (mfus_name, pfus_name)]  # expand mFus to pFus
        gap_name = f'{hemi2Hemi[hemi]}_FFA-gap'
        gap_key = roi2key[gap_name]

        # loop subjects
        for idx, lbl_tab in enumerate(lbl_tabs):
            time1 = time.time()
            if df.loc[idx, hemi] == 2:
                edge_mask = label_edge_detection(
                    data_hemi[idx], faces, 'inner', neighbors_list)
                roi2edge = {}
                roi2gdist_map = {}
                roi2lbl_map = {}
                for roi in (pfus_name, mfus_name):
                    roi2edge[roi] = np.where(
                        edge_mask == roi2key[roi])[0].astype(np.int32)
                    roi2gdist_map[roi] = gdist.compute_gdist(
                            coords, faces, roi2edge[roi], vertices)
                    roi2lbl_map[roi] = data_hemi[idx].copy()

                # expand ffa_pair[0] to ffa_pair[1]
                for ffa_pair in ffa_pairs:
                    lbl_map = roi2lbl_map[ffa_pair[0]]
                    base_vertices = roi2edge[ffa_pair[0]]
                    gdist_map = roi2gdist_map[ffa_pair[1]]
                    while len(base_vertices) > 0:
                        base_vertices_tmp = []
                        for base_vtx in base_vertices:
                            for neigh_vtx in neighbors_list[base_vtx]:
                                if lbl_map[neigh_vtx] != 0:
                                    continue
                                if gdist_map[neigh_vtx] < gdist_map[base_vtx]:
                                    lbl_map[neigh_vtx] = gap_key
                                    base_vertices_tmp.append(neigh_vtx)
                        base_vertices = np.array(base_vertices_tmp, np.int32)

                # get gap mask and assign gap key
                gap_mask = np.logical_and(roi2lbl_map[pfus_name] == gap_key,
                                          roi2lbl_map[mfus_name] == gap_key)
                if not np.any(gap_mask):
                    msg = f'No gap was found in {hemi}_{mns[idx][:6]}\n'
                    print(msg)
                    log_writer.write(msg)
                    continue
                data_hemi[idx][gap_mask] = gap_key
                lbl_tab[gap_key] = \
                    nib.cifti2.Cifti2Label(gap_key, gap_name, *gap_rgba)
                mns[idx] = mns[idx] + hemi[0]
            else:
                data_hemi[idx] = 0
                invalid_keys = []
                for k, v in lbl_tab.items():
                    assert roi2key[v.label] == k
                    if v.label.startswith(f'{hemi2Hemi[hemi]}_'):
                        invalid_keys.append(k)
                for k in invalid_keys:
                    lbl_tab.pop(k)
            print(f'Finished {idx+1}/{n_subj}, '
                  f'cost {time.time()-time1} seconds.')
        data.append(data_hemi[:, idx2vtx])
    log_writer.close()

    data = np.concatenate(data, axis=1)
    save2cifti(out_file, data, bms, mns, label_tables=lbl_tabs)


def mask_FFA_gap():
    """
    将FFA gap限制在FFC之内
    如果有FFA gap与FFC没有交集，则会从对应被试的map name中去掉半脑标记（l or r）
    以及去掉label table里的label。
    """
    # settings
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    src_file = pjoin(work_dir, 'FFA+gap1_indiv.32k_fs_LR.dlabel.nii')
    out_file = pjoin(work_dir, 'FFA+gap1-in-FFC_indiv.32k_fs_LR.dlabel.nii')
    log_file = pjoin(work_dir, 'FFA+gap1-in-FFC_indiv_log')
    roi2key = {'R_FFA-gap': 5, 'L_FFA-gap': 6}

    # loading
    reader = CiftiReader(src_file)
    data = reader.get_data()
    bms = reader.brain_models()
    mns = reader.map_names()
    lbl_tabs = reader.label_tables()
    mmp_map = nib.load(mmp_map_file).get_fdata()[0]

    # calculating
    log_writer = open(log_file, 'w')
    for hemi in hemis:
        ffc_mask = mmp_map == mmp_name2label[f'{hemi2Hemi[hemi]}_FFC']
        gap_key = roi2key[f'{hemi2Hemi[hemi]}_FFA-gap']
        for subj_idx, lbl_tab in enumerate(lbl_tabs):
            if hemi[0] not in mns[subj_idx]:
                continue
            gap_mask_old = data[subj_idx] == gap_key
            data[subj_idx][gap_mask_old] = 0
            gap_mask = np.logical_and(gap_mask_old, ffc_mask)
            if not np.any(gap_mask):
                msg = f'No gap was found in {hemi}_{mns[subj_idx][:6]}\n'
                print(msg)
                log_writer.write(msg)
                lbl_tab.pop(gap_key)
                mns[subj_idx] = mns[subj_idx].replace(hemi[0], '')
            else:
                data[subj_idx][gap_mask] = gap_key
    save2cifti(out_file, data, bms, mns, label_tables=lbl_tabs)


def thr_FFA_gap(thr):
    """
    将FFA gap限制在face-avg Z<thr之内
    如果有所有点都不在Z<thr的范围内的FFA gap，
    则会从对应被试的map name中去掉半脑标记（l or r）
    以及去掉label table里的label。
    """
    # settings
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    src_file = pjoin(work_dir, 'FFA+gap1-in-FFC_indiv.32k_fs_LR.dlabel.nii')
    activ_file = pjoin(anal_dir, 'NI_R1/data_1053/'
                       'S1200_1053_tfMRI_WM_level2_FACE-AVG_hp200_s2_MSMAll.32k_fs_LR.dscalar.nii')
    subj_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    out_file = pjoin(work_dir, f'FFA+gap1-in-FFC_thr{thr}_indiv.32k_fs_LR.dlabel.nii')
    log_file = pjoin(work_dir, f'FFA+gap1-in-FFC_thr{thr}_indiv_log')
    roi2key = {'R_FFA-gap': 5, 'L_FFA-gap': 6}

    # loading
    reader = CiftiReader(src_file)
    data = reader.get_data()
    bms = reader.brain_models()
    mns = reader.map_names()
    lbl_tabs = reader.label_tables()
    thr_masks = nib.load(activ_file).get_fdata() < thr
    subj_ids = open(subj_file).read().splitlines()

    # calculating
    log_writer = open(log_file, 'w')
    for hemi in hemis:
        gap_key = roi2key[f'{hemi2Hemi[hemi]}_FFA-gap']
        for subj_idx, lbl_tab in enumerate(lbl_tabs):
            if hemi[0] not in mns[subj_idx]:
                continue
            subj_id = mns[subj_idx][:6]
            thr_mask_idx = subj_ids.index(subj_id)
            thr_mask = thr_masks[thr_mask_idx]
            gap_mask_old = data[subj_idx] == gap_key
            data[subj_idx][gap_mask_old] = 0
            gap_mask = np.logical_and(gap_mask_old, thr_mask)
            if not np.any(gap_mask):
                msg = f'No gap was found in {hemi}_{subj_id}\n'
                print(msg)
                log_writer.write(msg)
                lbl_tab.pop(gap_key)
                mns[subj_idx] = mns[subj_idx].replace(hemi[0], '')
            else:
                data[subj_idx][gap_mask] = gap_key
    save2cifti(out_file, data, bms, mns, label_tables=lbl_tabs)


def prepare_mmp_series(sess=1, run='LR'):
    """
    为1053个被试提取360个HCP MMP脑区的静息时间序列
    """
    # settings
    n_tp = 1200
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    rfmri_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
                  'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    sid_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    out_dir = pjoin(work_dir, 'rfMRI')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = pjoin(out_dir, f'rfMRI_MMP_{sess}_{run}.pkl')

    # loading
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'
    sids = open(sid_file).read().splitlines()
    n_sid = len(sids)

    mmp_map = nib.load(mmp_map_file).get_fdata()[0]
    mmp_rois = list(mmp_name2label.keys())
    mmp_roi2mask = {}
    for mmp_roi in mmp_rois:
        mmp_roi2mask[mmp_roi] = mmp_map == mmp_name2label[mmp_roi]

    # calculating
    out_dict = {
        'subID': sids,
        'shape': 'n_subject x n_roi x n_time_point',
        'roi': mmp_rois,
        'data': np.ones((n_sid, len(mmp_rois), n_tp)) * np.nan
    }
    for sidx, sid in enumerate(sids):
        time1 = time.time()
        if not df_check.loc[int(sid), f'rfMRI_REST{sess}_{run}']:
            continue
        rfmri_file = rfmri_files.format(sid, sess, run)
        tseries = nib.load(rfmri_file).get_fdata()[:, :LR_count_32k]
        for mmp_roi_idx, mmp_roi in enumerate(mmp_rois):
            out_dict['data'][sidx, mmp_roi_idx] = \
                np.mean(tseries[:, mmp_roi2mask[mmp_roi]], 1)
        print(f'Finished {sess}-{run}-{sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def prepare_ffa_series(sess=1, run='LR'):
    """
    为1053个被试提取个体FFA的静息时间序列
    """
    # settings
    n_tp = 1200
    sid_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    ffa_names = ['R_pFus-faces', 'R_mFus-faces',
                 'L_pFus-faces', 'L_mFus-faces']
    ffa_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    rfmri_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
                  'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_dir = pjoin(work_dir, 'rfMRI')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = pjoin(out_dir, f'rfMRI_FFA_{sess}_{run}.pkl')

    # loading
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'
    sids = open(sid_file).read().splitlines()
    n_sid = len(sids)

    reader = CiftiReader(ffa_file)
    assert sids == reader.map_names()
    ffa_maps = reader.get_data()
    lbl_tabs = reader.label_tables()

    # calculating
    out_dict = {
        'subID': sids,
        'shape': 'n_subject x n_roi x n_time_point',
        'roi': ffa_names,
        'data': np.ones((n_sid, len(ffa_names), n_tp)) * np.nan}
    for sidx, sid in enumerate(sids):
        time1 = time.time()
        if not df_check.loc[int(sid), f'rfMRI_REST{sess}_{run}']:
            continue
        rfmri_file = rfmri_files.format(sid, sess, run)
        tseries = nib.load(rfmri_file).get_fdata()[:, :LR_count_32k]
        for ffa_key in np.unique(ffa_maps[sidx]):
            if ffa_key == 0:
                continue
            roi_idx = ffa_names.index(lbl_tabs[sidx][ffa_key].label)
            out_dict['data'][sidx, roi_idx] = \
                np.mean(tseries[:, ffa_maps[sidx] == ffa_key], 1)
        print(f'Finished {sess}-{run}-{sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def prepare_gap_series(sess=1, run='LR'):
    """
    为拥有gap的半脑提取gap的静息时间序列
    """
    # settings
    n_tp = 1200
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    gap_types = ('gap1-in-FFC', 'gap1-in-FFC_thr0.5', 'gap1-in-FFC_thr0')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    rfmri_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
                  'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_dir = pjoin(work_dir, 'rfMRI')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    gap_maps_list = []
    mns_list = []
    out_files = []
    out_dicts = []
    sids = None
    for gap_type in gap_types:
        gap_file = pjoin(work_dir, f'FFA+{gap_type}_indiv.32k_fs_LR.dlabel.nii')
        reader = CiftiReader(gap_file)
        gap_maps_list.append(reader.get_data())

        mns = reader.map_names()
        mns_list.append(mns)
        if sids is None:
            sids = [mn[:6] for mn in mns]
        else:
            assert sids == [mn[:6] for mn in mns]

        out_files.append(pjoin(out_dir, f'rfMRI_{gap_type}_{sess}_{run}.pkl'))
        out_dict = {}
        for hemi in hemis:
            sids_hemi = [mn[:6] for mn in mns if hemi[0] in mn]
            out_dict[hemi] = {
                'subID': sids_hemi,
                'shape': 'n_subject x n_time_point',
                'data': np.ones((len(sids_hemi), n_tp)) * np.nan}
        out_dicts.append(out_dict)
    n_sid = len(sids)

    roi2key = {'R_FFA-gap': 5, 'L_FFA-gap': 6}
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'

    for sidx, sid in enumerate(sids):
        time1 = time.time()
        if not df_check.loc[int(sid), f'rfMRI_REST{sess}_{run}']:
            continue
        rfmri_file = rfmri_files.format(sid, sess, run)
        tseries = nib.load(rfmri_file).get_fdata()[:, :LR_count_32k]
        for hemi in hemis:
            gap_key = roi2key[f'{hemi2Hemi[hemi]}_FFA-gap']
            for gap_idx, mns in enumerate(mns_list):
                if hemi[0] not in mns[sidx]:
                    continue
                gap_map = gap_maps_list[gap_idx][sidx]
                sidx_hemi = out_dicts[gap_idx][hemi]['subID'].index(sid)
                out_dicts[gap_idx][hemi]['data'][sidx_hemi] = \
                    np.mean(tseries[:, gap_map == gap_key], 1)
        print(f'Finished {sess}-{run}-{sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    for out_file, out_dict in zip(out_files, out_dicts):
        pkl.dump(out_dict, open(out_file, 'wb'))


def rsfc(sess=1, run='LR'):
    """
    为拥有gap的半脑计算gap，个体FFA，FFC
    以及FFC周边脑区与360个HCP MMP脑区的静息态功能连接
    """
    n_tp = 1200
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    gap_type = 'gap1-in-FFC_thr0'
    seeds = ['FFA-gap', 'pFus-faces', 'mFus-faces', 'FFC',
             'V8', 'PIT', 'PH', 'TE2p', 'TF', 'VVC']
    out_dir = pjoin(work_dir, 'rfMRI')
    gap_tfile = pjoin(out_dir, f'rfMRI_{gap_type}_{sess}_{run}.pkl')
    ffa_tfile = pjoin(out_dir, f'rfMRI_FFA_{sess}_{run}.pkl')
    mmp_tfile = pjoin(out_dir, f'rfMRI_MMP_{sess}_{run}.pkl')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(out_dir, f'rsfc_{gap_type}_{sess}_{run}.pkl')

    gap_tdata = pkl.load(open(gap_tfile, 'rb'))
    ffa_tdata = pkl.load(open(ffa_tfile, 'rb'))
    mmp_tdata = pkl.load(open(mmp_tfile, 'rb'))
    targets = mmp_tdata['roi']
    n_trg = len(targets)
    n_seed = len(seeds)
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'
    out_dict = {}
    for hemi in hemis:
        Hemi = hemi2Hemi[hemi]
        sids = gap_tdata[hemi]['subID']
        n_sid = len(sids)
        out_dict[hemi] = {
            'subID': sids, 'seed': seeds, 'target': targets,
            'shape': 'n_subject x n_seed x n_target',
            'data': np.ones((n_sid, n_seed, n_trg)) * np.nan
        }
        for sidx, sid in enumerate(sids):
            if not df_check.loc[int(sid), f'rfMRI_REST{sess}_{run}']:
                continue
            sidx_ffa = ffa_tdata['subID'].index(sid)
            sidx_mmp = mmp_tdata['subID'].index(sid)

            seed_tseries = np.zeros((n_seed, n_tp))
            for seed_idx, seed in enumerate(seeds):
                if seed == 'FFA-gap':
                    seed_tseries[seed_idx] = gap_tdata[hemi]['data'][sidx]
                elif seed in ('pFus-faces', 'mFus-faces'):
                    roi_idx = ffa_tdata['roi'].index(f'{Hemi}_{seed}')
                    seed_tseries[seed_idx] = \
                        ffa_tdata['data'][sidx_ffa, roi_idx]
                else:
                    roi_idx = mmp_tdata['roi'].index(f'{Hemi}_{seed}')
                    seed_tseries[seed_idx] = \
                        mmp_tdata['data'][sidx_mmp, roi_idx]
            trg_tseries = mmp_tdata['data'][sidx_mmp]
            out_dict[hemi]['data'][sidx] = \
                1 - cdist(seed_tseries, trg_tseries, 'correlation')
    pkl.dump(out_dict, open(out_file, 'wb'))


def rsfc_new(sess=1, run='LR'):
    """
    为拥有gap的半脑计算gap，个体FFA，FFC以及FFC周边脑区
    与gap，个体FFA，360个HCP MMP脑区的静息态功能连接
    """
    n_tp = 1200
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    gap_type = 'gap1-in-FFC'
    seeds = ['FFA-gap', 'pFus-faces', 'mFus-faces', 'FFC',
             'V8', 'PIT', 'PH', 'TE2p', 'TF', 'VVC']
    out_dir = pjoin(work_dir, 'rfMRI')
    gap_tfile = pjoin(out_dir, f'rfMRI_{gap_type}_{sess}_{run}.pkl')
    ffa_tfile = pjoin(out_dir, f'rfMRI_FFA_{sess}_{run}.pkl')
    mmp_tfile = pjoin(out_dir, f'rfMRI_MMP_{sess}_{run}.pkl')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(out_dir, f'rsfc_{gap_type}_{sess}_{run}_new.pkl')

    gap_tdata = pkl.load(open(gap_tfile, 'rb'))
    ffa_tdata = pkl.load(open(ffa_tfile, 'rb'))
    mmp_tdata = pkl.load(open(mmp_tfile, 'rb'))
    n_seed = len(seeds)
    df_check = pd.read_csv(check_file, sep='\t', index_col='subID')
    df_check = df_check == 'ok=(1200, 91282)'
    out_dict = {}
    for hemi in hemis:
        Hemi = hemi2Hemi[hemi]
        sids = gap_tdata[hemi]['subID']
        n_sid = len(sids)
        targets = [f'{Hemi}_{i}' for i in seeds[:3]] + mmp_tdata['roi']
        out_dict[hemi] = {
            'subID': sids, 'seed': seeds, 'target': targets,
            'shape': 'n_subject x n_seed x n_target',
            'data': np.ones((n_sid, n_seed, len(targets))) * np.nan
        }
        for sidx, sid in enumerate(sids):
            if not df_check.loc[int(sid), f'rfMRI_REST{sess}_{run}']:
                continue
            sidx_ffa = ffa_tdata['subID'].index(sid)
            sidx_mmp = mmp_tdata['subID'].index(sid)

            seed_tseries = np.zeros((n_seed, n_tp))
            for seed_idx, seed in enumerate(seeds):
                if seed == 'FFA-gap':
                    seed_tseries[seed_idx] = gap_tdata[hemi]['data'][sidx]
                elif seed in ('pFus-faces', 'mFus-faces'):
                    roi_idx = ffa_tdata['roi'].index(f'{Hemi}_{seed}')
                    seed_tseries[seed_idx] = \
                        ffa_tdata['data'][sidx_ffa, roi_idx]
                else:
                    roi_idx = mmp_tdata['roi'].index(f'{Hemi}_{seed}')
                    seed_tseries[seed_idx] = \
                        mmp_tdata['data'][sidx_mmp, roi_idx]

            trg_tseries = np.r_[seed_tseries[:3, :],
                                mmp_tdata['data'][sidx_mmp]]
            out_dict[hemi]['data'][sidx] = \
                1 - cdist(seed_tseries, trg_tseries, 'correlation')
    pkl.dump(out_dict, open(out_file, 'wb'))


def rsfc_mean_among_run():

    sess = (1, 2)
    runs = ('LR', 'RL')
    hemis = ('lh', 'rh')
    gap_type = 'gap1-in-FFC'
    out_dir = pjoin(work_dir, 'rfMRI')
    fpaths = pjoin(out_dir, 'rsfc_{gap_type}_{ses}_{run}_new.pkl')
    out_file = pjoin(out_dir, f'rsfc_{gap_type}_new.pkl')

    first_flag = True
    rsfc_dict = dict()
    for ses in sess:
        for run in runs:
            fpath = fpaths.format(gap_type=gap_type, ses=ses, run=run)
            tmp_rsfc = pkl.load(open(fpath, 'rb'))
            if first_flag:
                for hemi in hemis:
                    rsfc_dict[hemi] = {}
                    rsfc_dict[hemi]['subID'] = tmp_rsfc[hemi]['subID']
                    rsfc_dict[hemi]['seed'] = tmp_rsfc[hemi]['seed']
                    rsfc_dict[hemi]['target'] = tmp_rsfc[hemi]['target']
                    rsfc_dict[hemi]['shape'] = tmp_rsfc[hemi]['shape']
                    rsfc_dict[hemi]['data'] = [tmp_rsfc[hemi]['data']]
                first_flag = False
            else:
                for hemi in hemis:
                    assert rsfc_dict[hemi]['subID'] == tmp_rsfc[hemi]['subID']
                    assert rsfc_dict[hemi]['seed'] == tmp_rsfc[hemi]['seed']
                    assert rsfc_dict[hemi]['target'] == tmp_rsfc[hemi]['target']
                    assert rsfc_dict[hemi]['shape'] == tmp_rsfc[hemi]['shape']
                    rsfc_dict[hemi]['data'].append(tmp_rsfc[hemi]['data'])

    for hemi in hemis:
        rsfc_dict[hemi]['data'] = np.nanmean(rsfc_dict[hemi]['data'], 0)
    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def rsfc_merge_MMP():
    """
    用ColeAnticevicNetPartition将MMP合并成12个网络
    """
    hemis = ('lh', 'rh')
    out_dir = pjoin(work_dir, 'rfMRI')
    rsfc_file = pjoin(out_dir, 'rsfc_gap1-in-FFC_new.pkl')
    out_file = pjoin(out_dir, 'rsfc_trg-Cole_gap1-in-FFC.pkl')

    roi2net_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
                   'cortex_parcel_network_assignments.mat'

    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))
    roi2net = loadmat(roi2net_file)['netassignments'][:, 0]
    roi2net = np.r_[roi2net[180:], roi2net[:180]]
    net_labels = sorted(set(roi2net))
    n_net = len(net_labels)

    out_dict = {}
    for hemi in hemis:
        n_sid = len(rsfc_dict[hemi]['subID'])
        n_seed = len(rsfc_dict[hemi]['seed'])
        out_dict[hemi] = {
            'subID': rsfc_dict[hemi]['subID'],
            'seed': rsfc_dict[hemi]['seed'],
            'target': net_labels,
            'shape': rsfc_dict[hemi]['shape'],
            'data': np.ones((n_sid, n_seed, n_net)) * np.nan}
        for net_idx, net_lbl in enumerate(net_labels):
            mmp_labels = np.where(roi2net == net_lbl)[0] + 1
            mmp_names = [mmp_label2name[i] for i in mmp_labels]
            trg_indices = [rsfc_dict[hemi]['target'].index(i) for i in mmp_names]
            data = np.mean(rsfc_dict[hemi]['data'][:, :, trg_indices], 2)
            out_dict[hemi]['data'][:, :, net_idx] = data

    pkl.dump(out_dict, open(out_file, 'wb'))


def pre_ANOVA_rsfc_data():
    """
    3-way repeated ANOVA with hemisphere (LH, RH),
    region (pFus, gap, mFus), and network (12 networks)
    as factors
    """
    hemis = ('lh', 'rh')
    rois = ['pFus-faces', 'FFA-gap', 'mFus-faces']
    roi2name = {'pFus-faces': 'pFus', 'FFA-gap': 'gap',
                'mFus-faces': 'mFus'}
    net_labels = np.arange(1, 13)
    out_dir = pjoin(work_dir, 'rfMRI')
    data_file = pjoin(out_dir, 'rsfc_trg-Cole_gap1-in-FFC.pkl')
    out_file = pjoin(out_dir, 'rsfc_trg-Cole_gap1-in-FFC_pre-ANOVA.csv')

    data = pkl.load(open(data_file, 'rb'))

    # 找出左右脑同属于separate组的被试
    sids = set(data['lh']['subID']).intersection(data['rh']['subID'])
    sids = sorted(sids)
    print('#sid_lh:', len(data['lh']['subID']))
    print('#sid_rh:', len(data['rh']['subID']))
    print('#sid_lr:', len(sids))

    out_dict = {}
    for hemi in hemis:
        sub_indices = [data[hemi]['subID'].index(i) for i in sids]
        assert np.all(net_labels == data[hemi]['target'])
        data_hemi = data[hemi]['data'][sub_indices]
        for roi in rois:
            roi_idx = data[hemi]['seed'].index(roi)
            for net_idx, net_lbl in enumerate(net_labels):
                meas_vec = data_hemi[:, roi_idx, net_idx]
                out_dict[f'{hemi}_{roi2name[roi]}_{net_lbl}'] = meas_vec
    pd.DataFrame(out_dict).to_csv(out_file, index=False)


def check_HCPY_tfMRI():
    """
    对各被试所有任务做以下检查并记录到一个表格中：
    表格每一行是一个被试，对应被试号存在'subID'列，
    其它列记录的是各任务各cope的状态，这些列的命名规则是
    {task}_cope{cope_num}_{cope_name}；这些列中元素的赋值规则：
    0. 初始值为''
    1. 对应任务存在Contrasts.txt，增加字符'a'
        已经证明对于所有任务：all exist Contrasts.txt files
        can be loaded and have the same content
    2. 对应任务存在{sid}_tfMRI_{task}_level2_hp200_s2_MSMAll.dscalar.nii,
        并且数据可以顺利被加载，增加字符'b'；如果形状不是(n_cope, 91282)或
        contrasts != contrasts_tmp，改为字符'B'
    3. 存在zstat1.dtseries.nii，并且数据可以顺利被加载，增加字符'c'；
        如果形状不是(1, 91282)或与all_cope_data中对应map不一致，
        改为字符'C'
    4. 存在cope1.dtseries.nii，并且数据可以顺利被加载，增加字符'd';
        如果形状不是(1, 91282，改为字符'D'
    """
    info_file = '/nfs/m1/hcp/S1200_behavior.csv'
    # tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
    #          'RELATIONAL', 'SOCIAL', 'WM']
    tasks = ['WM']
    feat_dir = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/tfMRI_{task}/'\
        'tfMRI_{task}_hp200_s2_level2_MSMAll.feat'
    contrast_files = pjoin(feat_dir, 'Contrasts.txt')
    all_cope_files = pjoin(
        feat_dir, '{sid}_tfMRI_{task}_level2_hp200_s2_MSMAll.dscalar.nii')
    zstat_files = pjoin(
        feat_dir, 'GrayordinatesStats/cope{c_num}.feat/zstat1.dtseries.nii')
    cope_files = pjoin(
        feat_dir, 'GrayordinatesStats/cope{c_num}.feat/cope1.dtseries.nii')
    out_dir = pjoin(work_dir, 'tfMRI')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # out_file = pjoin(out_dir, 'HCPY_tfMRI_check.tsv')
    # log_file = pjoin(out_dir, 'HCPY_tfMRI_check_log')
    out_file = pjoin(out_dir, 'HCPY_tfMRI-WM_check.tsv')
    log_file = pjoin(out_dir, 'HCPY_tfMRI-WM_check_log')

    df = pd.read_csv(info_file, usecols=['Subject'])
    sids = df['Subject'].values
    n_sid = len(sids)
    out_dict = {'subID': sids}
    log_handle = open(log_file, 'w')

    task2contrasts = {}
    for task in tasks:
        contrast_text = None
        for sid in sids:
            contrast_file = contrast_files.format(sid=sid, task=task)
            if os.path.isfile(contrast_file):
                if contrast_text is None:
                    contrast_text = open(contrast_file).read()
                else:
                    assert contrast_text == open(contrast_file).read()
        task2contrasts[task] = contrast_text.splitlines()
        log_handle.write(f'task-{task}: all exist Contrasts.txt files '
                         'can be loaded and have the same content.\n')

    for task in tasks:
        contrasts = task2contrasts[task]
        n_cope = len(contrasts)
        for c_num, c_name in enumerate(contrasts, 1):
            out_dict[f'{task}_cope{c_num}_{c_name}'] = []
        for sidx, sid in enumerate(sids, 1):
            time1 = time.time()
            contrast_file = contrast_files.format(sid=sid, task=task)
            if os.path.isfile(contrast_file):
                status_code1 = 'a'
            else:
                status_code1 = ''

            all_cope_file = all_cope_files.format(sid=sid, task=task)
            try:
                reader = CiftiReader(all_cope_file)
                all_cope_data = reader.get_data()
                contrasts_tmp = ['_'.join(i.split('_')[4:-3])
                                 for i in reader.map_names()]
                if all_cope_data.shape == (n_cope, 91282) and contrasts == contrasts_tmp:
                    status_code1 += 'b'
                else:
                    status_code1 += 'B'
            except Exception as err:
                log_handle.write(f'Error in {task}-{sid}: {err}\n')

            for c_num, c_name in enumerate(contrasts, 1):
                status_code2 = status_code1

                zstat_file = zstat_files.format(
                    sid=sid, task=task, c_num=c_num)
                try:
                    zstat_map = nib.load(zstat_file).get_fdata()
                    if zstat_map.shape == (1, 91282):
                        if ('b' in status_code2) or ('B' in status_code2):
                            if np.all(all_cope_data[c_num - 1] == zstat_map[0]):
                                status_code2 += 'c'
                            else:
                                status_code2 += 'C'
                        else:
                            status_code2 += 'c'
                    else:
                        status_code2 += 'C'
                except Exception as err:
                    log_handle.write(f'Error in {task}-{sid}-{c_name}-zstat: {err}\n')

                cope_file = cope_files.format(
                    sid=sid, task=task, c_num=c_num)
                try:
                    cope_map = nib.load(cope_file).get_fdata()
                    if cope_map.shape == (1, 91282):
                        status_code2 += 'd'
                    else:
                        status_code2 += 'D'
                except Exception as err:
                    log_handle.write(f'Error in {task}-{sid}-{c_name}-cope: {err}\n')

                out_dict[f'{task}_cope{c_num}_{c_name}'].append(status_code2)
            print(f'Finish {task}-{sidx}/{n_sid}, '
                  f'cost {time.time()-time1} seconds')
    pd.DataFrame(out_dict).to_csv(out_file, index=False, sep='\t')


def get_cope_data(task='WM', gap_type='gap1-in-FFC'):
    """
    为拥有gap的半脑计算gap，个体FFA，FFC
    以及FFC周边脑区在"task"任务中感兴趣条件的激活值(beta)
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    gap_file = pjoin(work_dir, f'FFA+{gap_type}_indiv.32k_fs_LR.dlabel.nii')
    rois = ['FFA-gap', 'pFus-faces', 'mFus-faces', 'FFC', 'V8',
            'PIT', 'PH', 'TE2p', 'TF', 'VVC']
    feat_dir = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/tfMRI_{task}/'\
        'tfMRI_{task}_hp200_s2_level2_MSMAll.feat'
    contrast_file = pjoin(feat_dir.format(sid='100307', task=task),
                          'Contrasts.txt')
    cope_files = pjoin(
        feat_dir, 'GrayordinatesStats/cope{c_num}.feat/cope1.dtseries.nii')
    out_dir = pjoin(work_dir, 'tfMRI')
    out_file = pjoin(out_dir, f'tfMRI-{task}_{gap_type}.pkl')

    task2copes = {
        'WM': ['BODY', 'FACE', 'PLACE', 'TOOL',
               'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG']}
    copes = task2copes[task]
    contrasts = open(contrast_file).read().splitlines()
    c_nums = [contrasts.index(i) + 1 for i in copes]
    n_cope = len(copes)

    reader = CiftiReader(gap_file)
    gap_maps = reader.get_data()
    mns = reader.map_names()
    sids = [mn[:6] for mn in mns]
    n_sid = len(sids)

    n_roi = len(rois)
    out_dict = {}
    for hemi in hemis:
        sids_hemi = [mn[:6] for mn in mns if hemi[0] in mn]
        out_dict[hemi] = {
            'subID': sids_hemi, 'roi': rois, 'cope': copes,
            'shape': 'n_subject x n_roi x n_cope',
            'data': np.ones((len(sids_hemi), n_roi, n_cope)) * np.nan}

    roi2key = {'???': 0, 'R_pFus-faces': 1, 'R_mFus-faces': 2,
               'L_pFus-faces': 3, 'L_mFus-faces': 4,
               'R_FFA-gap': 5, 'L_FFA-gap': 6}
    mmp_map = nib.load(mmp_map_file).get_fdata()[0]
    for sidx, sid in enumerate(sids):
        time1 = time.time()
        gap_map = gap_maps[sidx]
        for cope_idx, c_num in enumerate(c_nums):
            cope_file = cope_files.format(sid=sid, task=task, c_num=c_num)
            try:
                cope_map = nib.load(cope_file).get_fdata()[0, :LR_count_32k]
            except Exception:
                continue
            for hemi in hemis:
                if hemi[0] not in mns[sidx]:
                    continue
                sidx_hemi = out_dict[hemi]['subID'].index(sid)
                for roi_idx, roi in enumerate(rois):
                    roi = f'{hemi2Hemi[hemi]}_{roi}'
                    if roi in roi2key.keys():
                        roi_mask = gap_map == roi2key[roi]
                    else:
                        roi_mask = mmp_map == mmp_name2label[roi]
                    out_dict[hemi]['data'][sidx_hemi, roi_idx, cope_idx] = \
                        np.mean(cope_map[roi_mask])
        print(f'Finished {task}-{gap_type}-{sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def get_cope_data_retest():
    """
    遍历所有test-retest被试，用在test被试上定出的gap，pFus，mFus在
    retest被试上提取WM任务的body，face，place，tool四个条件的激活值(beta)，
    得到3x4的矩阵。如果有被试不存在gap，则3x4的矩阵全为NAN，如果是缺失
    某个cope数据，则对应cope列为NAN。
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    sid_file = '/nfs/m1/hcp/retest/3T_tfMRI_WM_analysis_s2_ID'
    gap_file = pjoin(work_dir, 'FFA+gap1-in-FFC_indiv.32k_fs_LR.dlabel.nii')
    rois = ['FFA-gap', 'pFus-faces', 'mFus-faces']
    contrast_file = '/nfs/m1/hcp/100307/MNINonLinear/Results/tfMRI_WM/'\
        'tfMRI_WM_hp200_s2_level2_MSMAll.feat/Contrasts.txt'
    contrast_files = '/nfs/m1/hcp/retest/{sid}/MNINonLinear/Results/'\
        'tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat/Contrasts.txt'
    copes = ['BODY', 'FACE', 'PLACE', 'TOOL']
    cope_files = '/nfs/m1/hcp/retest/{sid}/MNINonLinear/Results/'\
        'tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat/'\
        'GrayordinatesStats/cope{c_num}.feat/cope1.dtseries.nii'
    out_dir = pjoin(work_dir, 'tfMRI')
    out_file = pjoin(out_dir, 'tfMRI-WM-retest_gap1-in-FFC.pkl')

    contrast_text = open(contrast_file).read()
    contrasts = contrast_text.splitlines()
    c_nums = [contrasts.index(i) + 1 for i in copes]
    n_cope = len(copes)

    reader = CiftiReader(gap_file)
    gap_maps = reader.get_data()
    mns = reader.map_names()
    gap_sids = [mn[:6] for mn in mns]
    sids = open(sid_file).read().splitlines()
    n_sid = len(sids)

    out_dict = {}
    for hemi in hemis:
        out_dict[hemi] = {
            'subID': sids, 'roi': rois, 'cope': copes,
            'shape': 'n_subject x n_roi x n_cope',
            'data': np.ones((n_sid, len(rois), n_cope)) * np.nan}

    roi2key = {'???': 0, 'R_pFus-faces': 1, 'R_mFus-faces': 2,
               'L_pFus-faces': 3, 'L_mFus-faces': 4,
               'R_FFA-gap': 5, 'L_FFA-gap': 6}
    for sidx, sid in enumerate(sids):
        time1 = time.time()
        if sid not in gap_sids:
            continue
        assert contrast_text == open(contrast_files.format(sid=sid)).read()
        gap_sidx = gap_sids.index(sid)
        gap_map = gap_maps[gap_sidx]
        for cope_idx, c_num in enumerate(c_nums):
            cope_file = cope_files.format(sid=sid, c_num=c_num)
            try:
                cope_map = nib.load(cope_file).get_fdata()
            except Exception:
                continue
            assert cope_map.shape == (1, 91282)
            cope_map = cope_map[0, :LR_count_32k]
            for hemi in hemis:
                if hemi[0] not in mns[gap_sidx]:
                    continue
                for roi_idx, roi in enumerate(rois):
                    roi = f'{hemi2Hemi[hemi]}_{roi}'
                    roi_mask = gap_map == roi2key[roi]
                    out_dict[hemi]['data'][sidx, roi_idx, cope_idx] = \
                        np.mean(cope_map[roi_mask])
        print(f'Finished {sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def pre_ANOVA_cope_data():
    """
    3-way repeated ANOVA with hemisphere (LH, RH),
    region (pFus, gap, mFus), and condition (face, body, place, tool)
    as factors
    """
    hemis = ('lh', 'rh')
    rois = ['pFus-faces', 'FFA-gap', 'mFus-faces']
    roi2name = {'pFus-faces': 'pFus', 'FFA-gap': 'gap',
                'mFus-faces': 'mFus'}
    conditions = ['FACE', 'BODY', 'PLACE', 'TOOL']
    sid_1053_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    gid_file = pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv')
    out_dir = pjoin(work_dir, 'tfMRI')
    data_file = pjoin(out_dir, 'tfMRI-WM-retest_gap1-in-FFC.pkl')
    out_file = pjoin(out_dir, 'tfMRI-WM-retest_gap1-in-FFC_pre-ANOVA.csv')

    data = pkl.load(open(data_file, 'rb'))

    # 找出左右脑同属于separate组的被试
    sid_1053 = open(sid_1053_file).read().splitlines()
    gid_df = pd.read_csv(gid_file)
    sid_45 = data['lh']['subID']
    assert sid_45 == data['rh']['subID']
    indices_in_1053 = [sid_1053.index(i) for i in sid_45]
    gid_df = gid_df.loc[indices_in_1053].reset_index(drop=True)
    hemi2g2_idx_vec = {
        'lh': gid_df['lh'] == 2,
        'rh': gid_df['rh'] == 2}
    g2_idx_vec = np.logical_and(hemi2g2_idx_vec['lh'], hemi2g2_idx_vec['rh'])
    print('#g2_lh:', np.sum(hemi2g2_idx_vec['lh']))
    print('#g2_rh:', np.sum(hemi2g2_idx_vec['rh']))
    print('#g2_lr:', np.sum(g2_idx_vec))

    out_dict = {}
    for hemi in hemis:
        data_hemi = data[hemi]['data'][g2_idx_vec]
        data1 = data[hemi]['data'][hemi2g2_idx_vec[hemi]]
        assert np.all(~np.isnan(data1))
        data2 = data[hemi]['data'][~hemi2g2_idx_vec[hemi]]
        assert np.all(np.isnan(data2))
        for roi in rois:
            roi_idx = data[hemi]['roi'].index(roi)
            for c_name in conditions:
                c_idx = data[hemi]['cope'].index(c_name)
                meas_vec = data_hemi[:, roi_idx, c_idx]
                out_dict[f'{hemi}_{roi2name[roi]}_{c_name}'] = meas_vec
    pd.DataFrame(out_dict).to_csv(out_file, index=False)


def get_stru_data(meas='myelin', gap_type='gap1-in-FFC'):
    """
    为拥有gap的半脑计算gap，个体FFA，FFC
    以及FFC周边脑区的myelin或thickness含量
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    gap_file = pjoin(work_dir, f'FFA+{gap_type}_indiv.32k_fs_LR.dlabel.nii')
    rois = ['FFA-gap', 'pFus-faces', 'mFus-faces', 'FFC', 'V8',
            'PIT', 'PH', 'TE2p', 'TF', 'VVC']
    meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'}
    out_dir = pjoin(work_dir, 'structure')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = pjoin(out_dir, f'{meas}_{gap_type}.pkl')

    reader = CiftiReader(gap_file)
    gap_maps = reader.get_data()
    mns = reader.map_names()
    sids = [mn[:6] for mn in mns]
    n_sid = len(sids)

    meas_ids = open(meas_id_file).read().splitlines()
    meas_maps = nib.load(meas2file[meas]).get_fdata()

    n_roi = len(rois)
    out_dict = {}
    for hemi in hemis:
        sids_hemi = [mn[:6] for mn in mns if hemi[0] in mn]
        out_dict[hemi] = {
            'subID': sids_hemi, 'roi': rois,
            'shape': 'n_subject x n_roi',
            'data': np.ones((len(sids_hemi), n_roi)) * np.nan}

    roi2key = {'???': 0, 'R_pFus-faces': 1, 'R_mFus-faces': 2,
               'L_pFus-faces': 3, 'L_mFus-faces': 4,
               'R_FFA-gap': 5, 'L_FFA-gap': 6}
    mmp_map = nib.load(mmp_map_file).get_fdata()[0]
    for sidx, sid in enumerate(sids):
        time1 = time.time()
        gap_map = gap_maps[sidx]
        meas_idx = meas_ids.index(sid)
        meas_map = meas_maps[meas_idx]
        for hemi in hemis:
            if hemi[0] not in mns[sidx]:
                continue
            sidx_hemi = out_dict[hemi]['subID'].index(sid)
            for roi_idx, roi in enumerate(rois):
                roi = f'{hemi2Hemi[hemi]}_{roi}'
                if roi in roi2key.keys():
                    roi_mask = gap_map == roi2key[roi]
                else:
                    roi_mask = mmp_map == mmp_name2label[roi]
                out_dict[hemi]['data'][sidx_hemi, roi_idx] = \
                    np.mean(meas_map[roi_mask])
        print(f'Finished {meas}-{gap_type}-{sidx+1}/{n_sid}, cost: '
              f'{time.time()-time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def pre_ANOVA_stru_data(meas='myelin'):
    """
    2-way repeated ANOVA with hemisphere (LH, RH),
    region (pFus, gap, mFus) as factors
    """
    hemis = ('lh', 'rh')
    rois = ['pFus-faces', 'FFA-gap', 'mFus-faces']
    roi2name = {'pFus-faces': 'pFus', 'FFA-gap': 'gap',
                'mFus-faces': 'mFus'}
    out_dir = pjoin(work_dir, 'structure')
    data_file = pjoin(out_dir, f'{meas}_gap1-in-FFC.pkl')
    out_file = pjoin(out_dir, f'{meas}_gap1-in-FFC_pre-ANOVA.csv')

    data = pkl.load(open(data_file, 'rb'))

    # 找出左右脑同属于separate组的被试
    sids = set(data['lh']['subID']).intersection(data['rh']['subID'])
    sids = sorted(sids)
    print('#sid_lh:', len(data['lh']['subID']))
    print('#sid_rh:', len(data['rh']['subID']))
    print('#sid_lr:', len(sids))

    out_dict = {}
    for hemi in hemis:
        sub_indices = [data[hemi]['subID'].index(i) for i in sids]
        data_hemi = data[hemi]['data'][sub_indices]
        for roi in rois:
            roi_idx = data[hemi]['roi'].index(roi)
            meas_vec = data_hemi[:, roi_idx]
            out_dict[f'{hemi}_{roi2name[roi]}'] = meas_vec
    pd.DataFrame(out_dict).to_csv(out_file, index=False)


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
    # zstat_corr_beta()
    # get_FFA_gap1()
    # mask_FFA_gap()
    # thr_FFA_gap(thr=0.5)
    # thr_FFA_gap(thr=0)

    # prepare_mmp_series(sess=1, run='LR')
    # prepare_mmp_series(sess=1, run='RL')
    # prepare_mmp_series(sess=2, run='LR')
    # prepare_mmp_series(sess=2, run='RL')
    # prepare_ffa_series(sess=1, run='LR')
    # prepare_ffa_series(sess=1, run='RL')
    # prepare_ffa_series(sess=2, run='LR')
    # prepare_ffa_series(sess=2, run='RL')
    # prepare_gap_series(sess=1, run='LR')
    # prepare_gap_series(sess=1, run='RL')
    # prepare_gap_series(sess=2, run='LR')
    # prepare_gap_series(sess=2, run='RL')
    # rsfc(sess=1, run='LR')
    # rsfc(sess=1, run='RL')
    # rsfc(sess=2, run='LR')
    # rsfc(sess=2, run='RL')
    # rsfc_mean_among_run()
    # rsfc_merge_MMP()
    pre_ANOVA_rsfc_data()

    # check_HCPY_tfMRI()
    # get_cope_data(task='WM', gap_type='gap1-in-FFC')
    # get_cope_data(task='WM', gap_type='gap1-in-FFC_thr0.5')
    # get_cope_data(task='WM', gap_type='gap1-in-FFC_thr0')
    # get_cope_data_retest()
    # pre_ANOVA_cope_data()

    # get_stru_data(meas='myelin', gap_type='gap1-in-FFC')
    # get_stru_data(meas='thickness', gap_type='gap1-in-FFC')
    # pre_ANOVA_stru_data(meas='myelin')
    # pre_ANOVA_stru_data(meas='thickness')

    # rsfc_new(sess=1, run='LR')
    # rsfc_new(sess=1, run='RL')
    # rsfc_new(sess=2, run='LR')
    # rsfc_new(sess=2, run='RL')
    # rsfc_mean_among_run()
