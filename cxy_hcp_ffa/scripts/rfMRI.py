import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from magicbox.io.io import CiftiReader

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'rfMRI')


# ---tools---
def get_name_label_of_ColeNetwork():
    import numpy as np

    src_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/' \
               'network_labelfile.txt'

    rf = open(src_file)
    names = []
    labels = []
    while True:
        name = rf.readline()
        if name == '':
            break
        names.append(name.rstrip('\n'))
        labels.append(int(rf.readline().split(' ')[0]))
    indices_sorted = np.argsort(labels)
    names = [names[i] for i in indices_sorted]
    labels = [labels[i] for i in indices_sorted]

    return names, labels


# ---steps---
def get_valid_id(sess=1, run='LR'):
    import os
    import time
    from magicbox.io.io import CiftiReader

    # parameters
    subj_id_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    maps_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
                 'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    # outputs
    log_file = pjoin(work_dir, f'get_valid_id_log_{sess}_{run}')
    trg_file = pjoin(work_dir, f'rfMRI_REST{sess}_{run}_id')

    subj_ids = open(subj_id_file).read().splitlines()
    n_subj = len(subj_ids)

    valid_ids = []
    log_writer = open(log_file, 'w')
    for idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()
        maps_file = maps_files.format(subj_id, sess, run)
        if not os.path.exists(maps_file):
            msg = f'{maps_file} is not exist.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        try:
            data = CiftiReader(maps_file).get_data()
        except OSError:
            msg = f'{maps_file} meets OSError.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        if data.shape[0] != 1200:
            msg = f'The number of time points in {maps_file} is not 1200.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        valid_ids.append(subj_id)
        print(f'Finished: {idx}/{n_subj}, cost: {time.time() - time1} seconds')
    log_writer.close()

    # save out
    with open(trg_file, 'w') as wf:
        wf.write('\n'.join(valid_ids))


def get_common_id():
    # parameters
    files = ['rfMRI_REST1_LR_id', 'rfMRI_REST1_RL_id',
             'rfMRI_REST2_LR_id', 'rfMRI_REST2_RL_id']
    files = [pjoin(work_dir, f) for f in files]

    # outputs
    trg_file = pjoin(work_dir, 'rfMRI_REST_id')

    ids = set(open(files[0]).read().splitlines())
    for f in files[1:]:
        ids.intersection_update(open(f).read().splitlines())
    ids = '\n'.join(sorted(ids))
    with open(trg_file, 'w') as wf:
        wf.write(ids)


def prepare_series_ind(sess=1, run='LR'):
    import numpy as np
    import nibabel as nib
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label
    from magicbox.io.io import CiftiReader

    # prepare subjects info
    subj_id_file = pjoin(work_dir, 'rfMRI_REST_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_ids = open(subj_id_file).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    subj_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids]
    n_subj = len(subj_ids)

    # prepare seeds
    seeds = ('IOG-face', 'pFus-face', 'mFus-face')
    n_seed = len(seeds)
    seed_mask_lh_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                              'refined_with_Kevin/rois_v3_lh.nii.gz')
    seed_mask_rh_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                              'refined_with_Kevin/rois_v3_rh.nii.gz')
    hemi2seed_mask = {
        'lh': nib.load(seed_mask_lh_file).get_fdata().squeeze().T[subj_idx_in_1080],
        'rh': nib.load(seed_mask_rh_file).get_fdata().squeeze().T[subj_idx_in_1080]}

    # prepare series dictionary
    n_tp = 1200  # the number of time points
    seed_lh_dict = {
        'shape': 'n_subject x n_seed x n_time_point',
        'subject': subj_ids,
        'seed': seeds,
        'rfMRI': np.ones((n_subj, n_seed, n_tp)) * np.nan
    }
    seed_rh_dict = {
        'shape': 'n_subject x n_seed x n_time_point',
        'subject': subj_ids,
        'seed': seeds,
        'rfMRI': np.ones((n_subj, n_seed, n_tp)) * np.nan
    }
    hemi2seed_dict = {
        'lh': seed_lh_dict,
        'rh': seed_rh_dict
    }

    # prepare outputs
    out_seed_lh = pjoin(work_dir, f'rfMRI{sess}_{run}_ROIv3_lh.pkl')
    out_seed_rh = pjoin(work_dir, f'rfMRI{sess}_{run}_ROIv3_rh.pkl')

    # start
    maps_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
                 'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    for subj_idx, subj_id in enumerate(subj_ids):
        print('Progress: {}/{}'.format(subj_idx+1, n_subj))

        # prepare maps
        maps_file = maps_files.format(subj_id, sess, run)
        maps_reader = CiftiReader(maps_file)
        hemi2maps = {
            'lh': maps_reader.get_data(hemi2stru['lh'], True),
            'rh': maps_reader.get_data(hemi2stru['rh'], True)}

        for hemi in ['lh', 'rh']:

            maps = hemi2maps[hemi]

            # seed dict
            seed_mask = hemi2seed_mask[hemi]
            for seed_idx, seed in enumerate(seeds):
                idx_vec = seed_mask[subj_idx] == roi2label[seed]
                if np.any(idx_vec):
                    hemi2seed_dict[hemi]['rfMRI'][subj_idx, seed_idx] =\
                        np.mean(maps[:, idx_vec], 1)

    # save
    pkl.dump(hemi2seed_dict['lh'], open(out_seed_lh, 'wb'))
    pkl.dump(hemi2seed_dict['rh'], open(out_seed_rh, 'wb'))


def prepare_series_mpm(sess=1, run='LR'):

    # prepare seeds
    seed2label = {
        'rh_pFus-face': 1, 'rh_mFus-face': 2,
        'lh_pFus-face': 3, 'lh_mFus-face': 4}
    seeds = ['pFus-face', 'mFus-face']
    n_seed = len(seeds)
    seed_mask_file = pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii')
    seed_map = nib.load(seed_mask_file).get_fdata()[0]
    n_vtx = seed_map.shape[0]

    # prepare series dictionary
    subj_id_file = pjoin(work_dir, 'rfMRI_REST_id')
    subj_ids = open(subj_id_file).read().splitlines()
    n_subj = len(subj_ids)
    n_tp = 1200  # the number of time points

    seed_lh_dict = {
        'shape': 'n_subject x n_seed x n_time_point',
        'subject': subj_ids,
        'seed': seeds,
        'rfMRI': np.ones((n_subj, n_seed, n_tp)) * np.nan
    }
    seed_rh_dict = {
        'shape': 'n_subject x n_seed x n_time_point',
        'subject': subj_ids,
        'seed': seeds,
        'rfMRI': np.ones((n_subj, n_seed, n_tp)) * np.nan
    }
    hemi2seed_dict = {
        'lh': seed_lh_dict,
        'rh': seed_rh_dict
    }
    out_lh = pjoin(work_dir, 'rfMRI_1053mpm_lh_{}_{}.pkl'.format(sess, run))
    out_rh = pjoin(work_dir, 'rfMRI_1053mpm_rh_{}_{}.pkl'.format(sess, run))

    # start
    maps_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/rfMRI_REST{1}_{2}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    for subj_idx, subj_id in enumerate(subj_ids):
        time1 = time.time()

        # prepare maps
        maps_file = maps_files.format(subj_id, sess, run)
        maps_reader = CiftiReader(maps_file)
        maps = maps_reader.get_data()[:, :n_vtx]
        for hemi in ['lh', 'rh']:
            for seed_idx, seed in enumerate(seeds):
                seed_mask = seed_map == seed2label[f'{hemi}_{seed}']
                hemi2seed_dict[hemi]['rfMRI'][subj_idx, seed_idx] = np.mean(maps[:, seed_mask], 1)
        print(f'Finished: {subj_idx+1}/{n_subj}, cost: {time.time()-time1}')

    pkl.dump(hemi2seed_dict['lh'], open(out_lh, 'wb'))
    pkl.dump(hemi2seed_dict['rh'], open(out_rh, 'wb'))


def rsfc(hemi='lh', sess=1, run='LR'):

    # parameters
    # seed_file = pjoin(work_dir, f'rfMRI{sess}_{run}_ROIv3_{hemi}.pkl')
    # out_file = pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_{sess}_{run}.pkl')

    seed_file = pjoin(work_dir, f'rfMRI_1053mpm_{hemi}_{sess}_{run}.pkl')
    out_file = pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}_{sess}_{run}.pkl')

    trg_file = pjoin(work_dir, f'rfMRI_trg_{sess}_{run}.pkl')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')

    # load data
    seed_dict = pkl.load(open(seed_file, 'rb'))
    trg_dict = pkl.load(open(trg_file, 'rb'))
    n_trg = len(trg_dict['trg_label'])
    assert seed_dict['subject'] == trg_dict['subject']
    subj_ids = open(subj_file).read().splitlines()

    # prepare FC dictionary
    fc_dict = {
        'shape': 'n_subject x n_target',
        'subject': subj_ids,
        'trg_label': trg_dict['trg_label']
    }
    for seed in seed_dict['seed']:
        fc_dict[seed] = np.ones((len(subj_ids), n_trg)) * np.nan

    # start
    subj_ids_valid = seed_dict['subject']
    n_valid = len(subj_ids_valid)
    for valid_idx, valid_id in enumerate(subj_ids_valid):
        print('Progress: {}/{}'.format(valid_idx+1, n_valid))
        subj_idx = subj_ids.index(valid_id)
        for seed_idx, seed in enumerate(seed_dict['seed']):
            seed_series = seed_dict['rfMRI'][valid_idx, [seed_idx]]
            if not np.isnan(seed_series[0, 0]):
                fc = 1 - cdist(seed_series, trg_dict['rfMRI'][valid_idx],
                               metric='correlation')[0]
                fc_dict[seed][subj_idx] = fc

    pkl.dump(fc_dict, open(out_file, 'wb'))


def fc_mean_among_run(hemi='lh'):
    # parameters
    # files = [
    #     pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_1_LR.pkl'),
    #     pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_1_RL.pkl'),
    #     pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_2_LR.pkl'),
    #     pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_2_RL.pkl'),
    # ]
    # rois = ['IOG-face', 'pFus-face', 'mFus-face']
    # out_file = pjoin(work_dir, f'rsfc_individual2MMP_{hemi}.pkl')

    files = [
        pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}_1_LR.pkl'),
        pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}_1_RL.pkl'),
        pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}_2_LR.pkl'),
        pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}_2_RL.pkl'),
    ]
    rois = ['pFus-face', 'mFus-face']
    out_file = pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}.pkl')

    rsfc_dict = dict()
    for idx, f in enumerate(files):
        tmp_rsfc = pkl.load(open(f, 'rb'))
        if idx == 0:
            rsfc_dict['shape'] = tmp_rsfc['shape']
            rsfc_dict['subject'] = tmp_rsfc['subject']
            rsfc_dict['trg_label'] = tmp_rsfc['trg_label']
            for roi in rois:
                rsfc_dict[roi] = [tmp_rsfc[roi]]
        else:
            assert rsfc_dict['shape'] == tmp_rsfc['shape']
            assert rsfc_dict['subject'] == tmp_rsfc['subject']
            assert np.all(rsfc_dict['trg_label'] == tmp_rsfc['trg_label'])
            for roi in rois:
                rsfc_dict[roi].append(tmp_rsfc[roi])
    for roi in rois:
        rsfc_dict[roi] = np.mean(rsfc_dict[roi], 0)

    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def fc_merge_MMP(hemi='lh'):
    """
    用ColeAnticevicNetPartition将MMP合并成12个网络
    """
    # parameters
    # seeds = ('IOG-face', 'pFus-face', 'mFus-face')
    # rsfc_file = pjoin(work_dir, f'rsfc_individual2MMP_{hemi}.pkl')
    # out_file = pjoin(work_dir, f'rsfc_individual2Cole_{hemi}.pkl')

    seeds = ('pFus-face', 'mFus-face')
    rsfc_file = pjoin(work_dir, f'rsfc_1053mpm2MMP_{hemi}.pkl')
    out_file = pjoin(work_dir, f'rsfc_1053mpm2Cole_{hemi}.pkl')

    roi2net_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
                   'cortex_parcel_network_assignments.mat'

    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))
    roi2net = loadmat(roi2net_file)['netassignments'][:, 0]
    roi2net = np.r_[roi2net[180:], roi2net[:180]]
    net_labels = sorted(set(roi2net))
    n_net = len(net_labels)

    for seed in seeds:
        data = np.zeros((rsfc_dict[seed].shape[0], n_net))
        for net_idx, net_lbl in enumerate(net_labels):
            data[:, net_idx] = np.mean(rsfc_dict[seed][:, roi2net == net_lbl], 1)
        rsfc_dict[seed] = data
    rsfc_dict['trg_label'] = net_labels

    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def pkl2mat(lh_file, rh_file, out_file, seeds, exclude_trg_labels=None):
    """
    把原来左右脑用两个pickle文件分开存的数据格式，转换到一个csv文件中

    Args:
        lh_file (str): 左脑pickle数据
        rh_file (str): 右脑pickle数据
        out_file (str): 整理后的CSV文件
        seeds (sequence): 指定整理哪些seeds
            IOG-face, pFus-face, mFus-face
        exclude_trg_labels (sequence, optional): Defaults to None.
            去掉和指定targets的连接
    """
    import numpy as np
    import pickle as pkl
    from scipy.io import savemat

    hemis = ('lh', 'rh')
    hemi2data = {
        'lh': pkl.load(open(lh_file, 'rb')),
        'rh': pkl.load(open(rh_file, 'rb'))
    }
    assert hemi2data['lh']['trg_label'] == hemi2data['rh']['trg_label']
    trg_labels = hemi2data['lh']['trg_label']

    if exclude_trg_labels is not None:
        exclude_trg_indices = [trg_labels.index(i) for i in exclude_trg_labels]
        for hemi in hemis:
            for seed in seeds:
                hemi2data[hemi][seed] = np.delete(hemi2data[hemi][seed],
                                                  exclude_trg_indices, 1)
        trg_labels = [i for i in trg_labels if i not in exclude_trg_labels]

    out_dict = {'target_label': trg_labels}
    for hemi in hemis:
        data = hemi2data[hemi]
        for seed in seeds:
            col = f"{hemi}_{seed.split('-')[0]}"
            out_dict[col] = data[seed]

    savemat(out_file, out_dict)


def fc_merge_Cole():
    """
    为每个被试每个ROI计算跨Cole networks的平均RSFC
    """
    rois = ['lh_pFus', 'lh_mFus', 'rh_pFus', 'rh_mFus']
    data = loadmat(pjoin(work_dir, 'rsfc_FFA2Cole.mat'))
    out_file = pjoin(work_dir, 'rsfc_FFA2Cole-mean.csv')
    out_data = {}
    for roi in rois:
        nan_arr = np.isnan(data[roi])
        assert np.all(np.all(nan_arr, 1) == np.any(nan_arr, 1))
        out_data[roi] = np.mean(data[roi], 1)
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_file, index=False)


def pre_ANOVA_rm():
    """
    Preparation for two-way repeated-measures ANOVA
    半球x脑区
    If use individual ROIs, only the subjects who have
    all four ROIs will be used.
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'rsfc_individual2Cole_{}.pkl')

    # outputs
    trg_file = pjoin(work_dir, 'rsfc_individual2Cole_preANOVA-rm.csv')

    out_dict = {}
    nan_idx_vec = np.zeros(1080, dtype=bool)
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi), 'rb'))
        for roi in rois:
            meas_vec = np.mean(data[roi], axis=1)
            nan_idx_vec = np.logical_or(nan_idx_vec, np.isnan(meas_vec))
            out_dict[f"{hemi}_{roi.split('-')[0]}"] = meas_vec
    valid_idx_vec = ~nan_idx_vec

    for k, v in out_dict.items():
        out_dict[k] = v[valid_idx_vec]
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def roi_ttest():
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pickle as pkl
    import pandas as pd
    from scipy.stats.stats import ttest_ind
    from cxy_hcp_ffa.lib.predefine import net2label_cole
    from magicbox.stats import EffectSize

    # parameters
    hemis = ('lh', 'rh')
    roi_pair = ('pFus-face', 'mFus-face')
    data_file = pjoin(work_dir, 'rsfc_individual2Cole_{}.pkl')
    compare_name = f"{roi_pair[0].split('-')[0]}_vs_" \
                   f"{roi_pair[1].split('-')[0]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_individual2Cole_{compare_name}_ttest.csv")

    # start
    trg_names = list(net2label_cole.keys())
    trg_labels = list(net2label_cole.values())
    out_data = {'network': trg_names}
    es = EffectSize()
    for hemi in hemis:
        data = pkl.load(open(data_file.format(hemi), 'rb'))
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


def roi_pair_ttest():
    """
    compare rsfc difference between ROIs
    scheme: hemi-separately network-wise
    """
    import numpy as np
    import pickle as pkl
    import pandas as pd
    from scipy.stats.stats import ttest_rel
    from cxy_hcp_ffa.lib.predefine import net2label_cole
    from magicbox.stats import EffectSize

    # inputs
    hemis = ('lh', 'rh')
    roi_pair = ('pFus-face', 'mFus-face')
    data_file = pjoin(work_dir, 'rsfc_individual2Cole_{}.pkl')
    compare_name = f"{roi_pair[0].split('-')[0]}_vs_" \
                   f"{roi_pair[1].split('-')[0]}"

    # outputs
    out_file = pjoin(work_dir,
                     f"rsfc_individual2Cole_{compare_name}_ttest_paired.csv")

    # start
    trg_names = list(net2label_cole.keys())
    trg_labels = list(net2label_cole.values())
    out_data = {'network': trg_names}
    es = EffectSize()
    for hemi in hemis:
        data = pkl.load(open(data_file.format(hemi), 'rb'))
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


def multitest_correct_ttest(fname='rsfc_individual2Cole_pFus_vs_mFus_ttest.csv'):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    # inputs
    hemis = ('lh', 'rh')
    data_file = pjoin(work_dir, fname)

    # outputs
    out_file = pjoin(work_dir, f"{fname.rstrip('.csv')}_mtc.csv")

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


def plot_bar():
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    # inputs
    hemis = ('lh', 'rh')
    seeds = ('pFus-face', 'mFus-face')
    seed2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    rsfc_file = pjoin(work_dir, 'rsfc_mpm2Cole_{hemi}.pkl')

    # outputs
    out_file = pjoin(work_dir, 'bar_mpm.jpg')

    n_hemi = len(hemis)
    n_seed = len(seeds)
    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_seed)
    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    hemi2meas = {
        'lh': pkl.load(open(rsfc_file.format(hemi='lh'), 'rb')),
        'rh': pkl.load(open(rsfc_file.format(hemi='rh'), 'rb'))}

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
    plt.savefig(out_file)
    # plt.show()


def prepare_plot(hemi='lh'):
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from cxy_hcp_ffa.lib.predefine import net2label_cole

    # inputs
    data_file = pjoin(work_dir, f'rsfc_individual2Cole_{hemi}.pkl')

    # outputs
    out_file = pjoin(work_dir, f'plot_rsfc_individual2Cole_{hemi}.pkl')

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


def plot_radar():
    """
    https://www.pythoncharts.com/2019/04/16/radar-charts/
    """
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt

    # inputs
    hemis = ('lh', 'rh')
    n_hemi = len(hemis)
    seed_names = ['pFus-face', 'mFus-face']
    seed2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    data_file = pjoin(work_dir, 'plot_rsfc_mpm2Cole_{hemi}.pkl')

    # outputs
    out_file = pjoin(work_dir, 'radar_mpm.jpg')

    trg_names = None
    trg_labels = None
    _, axes = plt.subplots(1, n_hemi, subplot_kw=dict(polar=True),
                           figsize=(6.4, 4.8))
    for hemi_idx, hemi in enumerate(hemis):
        ax = axes[hemi_idx]
        data = pkl.load(open(data_file.format(hemi=hemi), 'rb'))

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
    plt.savefig(out_file)

    for lbl, name in zip(trg_labels, trg_names):
        print(lbl, name)


if __name__ == '__main__':
    # get_valid_id(1, 'LR')
    # get_valid_id(1, 'RL')
    # get_valid_id(2, 'LR')
    # get_valid_id(2, 'RL')
    # get_common_id()
    # prepare_series_ind(sess=1, run='LR')
    # prepare_series_ind(sess=1, run='RL')
    # prepare_series_ind(sess=2, run='LR')
    # prepare_series_ind(sess=2, run='RL')
    # prepare_series_mpm(sess=1, run='LR')
    # prepare_series_mpm(sess=1, run='RL')
    # prepare_series_mpm(sess=2, run='LR')
    # prepare_series_mpm(sess=2, run='RL')
    # rsfc(hemi='lh', sess=1, run='LR')
    # rsfc(hemi='lh', sess=1, run='RL')
    # rsfc(hemi='lh', sess=2, run='LR')
    # rsfc(hemi='lh', sess=2, run='RL')
    # rsfc(hemi='rh', sess=1, run='LR')
    # rsfc(hemi='rh', sess=1, run='RL')
    # rsfc(hemi='rh', sess=2, run='LR')
    # rsfc(hemi='rh', sess=2, run='RL')
    # fc_mean_among_run(hemi='lh')
    # fc_mean_among_run(hemi='rh')
    fc_merge_MMP(hemi='lh')
    fc_merge_MMP(hemi='rh')
    # pkl2mat(
    #     lh_file=pjoin(work_dir, 'rsfc_individual2Cole_lh.pkl'),
    #     rh_file=pjoin(work_dir, 'rsfc_individual2Cole_rh.pkl'),
    #     out_file=pjoin(work_dir, 'rsfc_FFA2Cole.mat'),
    #     seeds=('pFus-face', 'mFus-face')
    # )
    # pkl2mat(
    #     lh_file=pjoin(work_dir, 'rsfc_individual2MMP_lh.pkl'),
    #     rh_file=pjoin(work_dir, 'rsfc_individual2MMP_rh.pkl'),
    #     out_file=pjoin(work_dir, 'rsfc_FFA2MMP.mat'),
    #     seeds=('pFus-face', 'mFus-face'),
    #     exclude_trg_labels=(18, 198)
    # )
    # fc_merge_Cole()
    # pre_ANOVA_rm()
    # roi_ttest()
    # multitest_correct_ttest(fname='rsfc_individual2Cole_pFus_vs_mFus_ttest.csv')
    # roi_pair_ttest()
    # multitest_correct_ttest(fname='rsfc_individual2Cole_pFus_vs_mFus_ttest_paired.csv')
    # prepare_plot(hemi='lh')
    # prepare_plot(hemi='rh')
    # plot_bar()
    # plot_radar()
