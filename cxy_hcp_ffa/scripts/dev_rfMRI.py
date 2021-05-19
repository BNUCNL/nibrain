from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
dev_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                'refined_with_Kevin/development')
work_dir = pjoin(dev_dir, 'rfMRI')


def get_valid_id(sess=1, run='AP'):
    import os
    import time
    import pandas as pd
    from magicbox.io.io import CiftiReader

    # inputs
    subj_info_file = pjoin(dev_dir, 'HCPD_SubjInfo.csv')

    # outputs
    log_file = pjoin(work_dir, f'get_valid_id_log_{sess}_{run}')
    out_file = pjoin(work_dir, f'rfMRI_REST{sess}_{run}_id')

    subj_info = pd.read_csv(subj_info_file)
    subj_ids = subj_info['subID'].to_list()
    n_subj = len(subj_ids)
    maps_files = '/nfs/e1/HCPD/fmriresults01/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/rfMRI_REST{1}_{2}_Atlas_MSMAll_hp0_clean.dtseries.nii'
    valid_ids = []
    log_writer = open(log_file, 'w')
    for idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()
        maps_file = maps_files.format(f'{subj_id}_V1_MR', sess, run)
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
        if data.shape[0] != 478:
            msg = f'The number of time points in {maps_file} is not 478.'
            print(msg)
            log_writer.write(f'{msg}\n')
            continue
        valid_ids.append(subj_id)
        print(f'Finished: {idx}/{n_subj}, cost: {time.time() - time1} seconds.')
    log_writer.close()

    # save out
    with open(out_file, 'w') as wf:
        wf.write('\n'.join(valid_ids))


def get_common_id():

    # inputs
    fnames = ['rfMRI_REST1_AP_id', 'rfMRI_REST1_PA_id',
              'rfMRI_REST2_AP_id', 'rfMRI_REST2_PA_id']
    fpaths = [pjoin(work_dir, fname) for fname in fnames]

    # outputs
    out_file = pjoin(work_dir, 'rfMRI_REST_4run_id')

    # calculate
    ids = set(open(fpaths[0]).read().splitlines())
    for f in fpaths[1:]:
        ids.intersection_update(open(f).read().splitlines())
    ids = '\n'.join(sorted(ids))

    # save
    with open(out_file, 'w') as wf:
        wf.write(ids)


def prepare_series(sess=1, run='AP'):
    import numpy as np
    import nibabel as nib
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label
    from magicbox.io.io import CiftiReader

    # prepare seeds
    seeds = ('IOG-face', 'pFus-face', 'mFus-face')
    n_seed = len(seeds)
    seed_mask_lh_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                              'refined_with_Kevin/MPM_v3_lh_0.25.nii.gz')
    seed_mask_rh_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                              'refined_with_Kevin/MPM_v3_rh_0.25.nii.gz')
    hemi2seed_mask = {
        'lh': nib.load(seed_mask_lh_file).get_fdata().squeeze(),
        'rh': nib.load(seed_mask_rh_file).get_fdata().squeeze()}

    # prepare targets
    trg_file = '/nfs/p1/atlases/multimodal_glasser/surface/'\
        'MMP_mpmLR32k.dlabel.nii'
    trg_reader = CiftiReader(trg_file)
    hemi2trg_mask = {
        'lh': trg_reader.get_data(hemi2stru['lh'], True).squeeze(),
        'rh': trg_reader.get_data(hemi2stru['rh'], True).squeeze()}

    trg_labels = np.unique(np.r_[hemi2trg_mask['lh'], hemi2trg_mask['rh']])
    trg_labels = trg_labels[trg_labels != 0].astype(int).tolist()
    n_trg = len(trg_labels)

    # prepare series dictionary
    subj_id_file = pjoin(work_dir, 'rfMRI_REST_4run_id')
    subj_ids = open(subj_id_file).read().splitlines()
    n_subj = len(subj_ids)
    n_tp = 478  # the number of time points

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
    trg_dict = {
        'shape': 'n_subject x n_target x n_time_point',
        'subject': subj_ids,
        'trg_label': trg_labels,
        'rfMRI': np.ones((n_subj, n_trg, n_tp)) * np.nan
    }
    global_dict = {
        'shape': 'n_subject x n_global x n_time_point',
        'subject': subj_ids,
        'global': ['lr', 'lh', 'rh'],
        'rfMRI': np.ones((n_subj, 3, n_tp)) * np.nan
    }

    # prepare outputs
    out_seed_lh = pjoin(work_dir, f'rfMRI{sess}_{run}_MPM_lh.pkl')
    out_seed_rh = pjoin(work_dir, f'rfMRI{sess}_{run}_MPM_rh.pkl')
    out_trg = pjoin(work_dir, f'rfMRI{sess}_{run}_MMP.pkl')
    out_global = pjoin(work_dir, f'rfMRI{sess}_{run}_global.pkl')

    # start
    maps_files = '/nfs/e1/HCPD/fmriresults01/{0}/MNINonLinear/'\
        'Results/rfMRI_REST{1}_{2}/'\
        'rfMRI_REST{1}_{2}_Atlas_MSMAll_hp0_clean.dtseries.nii'
    for subj_idx, subj_id in enumerate(subj_ids):
        print('Progress: {}/{}'.format(subj_idx+1, n_subj))

        # prepare maps
        maps_file = maps_files.format(f'{subj_id}_V1_MR', sess, run)
        maps_reader = CiftiReader(maps_file)
        hemi2maps = {
            'lh': maps_reader.get_data(hemi2stru['lh'], True),
            'rh': maps_reader.get_data(hemi2stru['rh'], True)}

        for hemi in ['lh', 'rh']:

            maps = hemi2maps[hemi]

            # seed dict
            seed_mask = hemi2seed_mask[hemi]
            for seed_idx, seed in enumerate(seeds):
                lbl = roi2label[seed]
                hemi2seed_dict[hemi]['rfMRI'][subj_idx, seed_idx] =\
                    np.mean(maps[:, seed_mask == lbl], 1)

            # target dict
            trg_mask = hemi2trg_mask[hemi]
            tmp_labels = np.unique(trg_mask)
            tmp_labels = tmp_labels[tmp_labels != 0].astype(int)
            for lbl in tmp_labels:
                trg_idx = trg_labels.index(lbl)
                trg_dict['rfMRI'][subj_idx, trg_idx] =\
                    np.mean(maps[:, trg_mask == lbl], 1)

            # global dict
            global_idx = global_dict['global'].index(hemi)
            global_dict['rfMRI'][subj_idx, global_idx] = np.mean(maps, 1)

        maps = np.c_[hemi2maps['lh'], hemi2maps['rh']]
        global_idx = global_dict['global'].index('lr')
        global_dict['rfMRI'][subj_idx, global_idx] = np.mean(maps, 1)
        del maps

    pkl.dump(hemi2seed_dict['lh'], open(out_seed_lh, 'wb'))
    pkl.dump(hemi2seed_dict['rh'], open(out_seed_rh, 'wb'))
    pkl.dump(trg_dict, open(out_trg, 'wb'))
    pkl.dump(global_dict, open(out_global, 'wb'))


def rsfc(sess=1, run='AP', hemi='lh'):
    """
    计算静息态功能连接
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl
    from scipy.spatial.distance import cdist

    # prepare seeds
    seed_file = pjoin(work_dir, f'rfMRI{sess}_{run}_MPM_{hemi}.pkl')
    seed_dict = pkl.load(open(seed_file, 'rb'))

    # prepare targets
    trg_file = pjoin(work_dir, f'rfMRI{sess}_{run}_MMP.pkl')
    trg_dict = pkl.load(open(trg_file, 'rb'))
    n_trg = len(trg_dict['trg_label'])

    assert seed_dict['subject'] == trg_dict['subject']

    # prepare FC dictionary
    subj_info_file = pjoin(dev_dir, 'HCPD_SubjInfo.csv')
    subj_ids = pd.read_csv(subj_info_file)['subID'].to_list()
    n_subj = len(subj_ids)
    fc_dict = {
        'shape': 'n_subject x n_target',
        'subject': subj_ids,
        'trg_label': trg_dict['trg_label']}
    for seed in seed_dict['seed']:
        fc_dict[seed] = np.ones((n_subj, n_trg)) * np.nan

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

    # save
    out_file = pjoin(work_dir, f'rsfc_MPM2MMP_{hemi}_{sess}_{run}.pkl')
    pkl.dump(fc_dict, open(out_file, 'wb'))


def rsfc_mean_among_run(hemi='lh'):
    import numpy as np
    import pickle as pkl

    # inputs
    fnames = [f'rsfc_MPM2MMP_{hemi}_1_AP.pkl', f'rsfc_MPM2MMP_{hemi}_1_PA.pkl',
              f'rsfc_MPM2MMP_{hemi}_2_AP.pkl', f'rsfc_MPM2MMP_{hemi}_2_PA.pkl']
    fpaths = [pjoin(work_dir, fname) for fname in fnames]
    rois = ['IOG-face', 'pFus-face', 'mFus-face']

    # outputs
    out_file = pjoin(work_dir, f'rsfc_MPM2MMP_{hemi}.pkl')

    # calculate
    rsfc_dict = dict()
    for idx, f in enumerate(fpaths):
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
            assert rsfc_dict['trg_label'] == tmp_rsfc['trg_label']
            for roi in rois:
                rsfc_dict[roi].append(tmp_rsfc[roi])
    for roi in rois:
        rsfc_dict[roi] = np.mean(rsfc_dict[roi], 0)

    # save
    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def rsfc_merge_MMP(hemi='lh'):
    """
    用ColeAnticevicNetPartition将MMP合并成12个网络
    """
    import numpy as np
    import pickle as pkl
    from scipy.io import loadmat

    rsfc_file = pjoin(work_dir, f'rsfc_MPM2MMP_{hemi}.pkl')
    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))

    roi2net_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
        'cortex_parcel_network_assignments.mat'
    roi2net = loadmat(roi2net_file)['netassignments'][:, 0]
    roi2net = np.r_[roi2net[180:], roi2net[:180]]
    net_labels = sorted(set(roi2net))
    n_net = len(net_labels)

    seeds = ('IOG-face', 'pFus-face', 'mFus-face')
    for seed in seeds:
        data = np.zeros((rsfc_dict[seed].shape[0], n_net))
        for net_idx, net_lbl in enumerate(net_labels):
            data[:, net_idx] = np.mean(rsfc_dict[seed][:, roi2net == net_lbl], 1)
        rsfc_dict[seed] = data
    rsfc_dict['trg_label'] = net_labels

    out_file = pjoin(work_dir, f'rsfc_MPM2Cole_{hemi}.pkl')
    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def plot_rsfc_line(hemi='lh'):
    """
    对于每个ROI，在每个年龄，求出ROI和targets连接的均值的
    被试间均值和SEM并画折线图
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl
    from scipy.stats.stats import sem
    from matplotlib import pyplot as plt
    from cxy_hcp_ffa.lib.predefine import roi2color

    # inputs
    rois = ('pFus-face', 'mFus-face')
    subj_info_file = pjoin(dev_dir, 'HCPD_SubjInfo.csv')
    rsfc_file = pjoin(work_dir, f'rsfc_MPM2Cole_{hemi}.pkl')

    # load
    subj_info = pd.read_csv(subj_info_file)
    age_vec = np.array(subj_info['age in years'])
    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))

    # plot
    for roi in rois:
        fc_vec = np.mean(rsfc_dict[roi], 1)
        non_nan_vec = ~np.isnan(fc_vec)
        fcs = fc_vec[non_nan_vec]
        ages = age_vec[non_nan_vec]
        age_uniq = np.unique(ages)
        ys = np.zeros_like(age_uniq, np.float64)
        yerrs = np.zeros_like(age_uniq, np.float64)
        for idx, age in enumerate(age_uniq):
            sample = fcs[ages == age]
            ys[idx] = np.mean(sample)
            yerrs[idx] = sem(sample)
        plt.errorbar(age_uniq, ys, yerrs,
                     label=roi, color=roi2color[roi])
    plt.ylabel('RSFC')
    plt.xlabel('age in years')
    plt.title(hemi)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # get_valid_id(sess=1, run='AP')
    # get_valid_id(sess=1, run='PA')
    # get_valid_id(sess=2, run='AP')
    # get_valid_id(sess=2, run='PA')
    # get_common_id()
    # prepare_series(sess=1, run='AP')
    # prepare_series(sess=1, run='PA')
    # prepare_series(sess=2, run='AP')
    # prepare_series(sess=2, run='PA')
    # rsfc(sess=1, run='AP', hemi='lh')
    # rsfc(sess=1, run='AP', hemi='rh')
    # rsfc(sess=1, run='PA', hemi='lh')
    # rsfc(sess=1, run='PA', hemi='rh')
    # rsfc(sess=2, run='AP', hemi='lh')
    # rsfc(sess=2, run='AP', hemi='rh')
    # rsfc(sess=2, run='PA', hemi='lh')
    # rsfc(sess=2, run='PA', hemi='rh')
    # rsfc_mean_among_run(hemi='lh')
    # rsfc_mean_among_run(hemi='rh')
    # rsfc_merge_MMP(hemi='lh')
    # rsfc_merge_MMP(hemi='rh')
    plot_rsfc_line(hemi='lh')
    plot_rsfc_line(hemi='rh')
