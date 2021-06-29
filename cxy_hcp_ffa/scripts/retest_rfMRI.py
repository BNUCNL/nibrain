from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                           'retest/rfMRI')


def get_valid_id(sess=1, run='LR'):
    import os
    import time
    import nibabel as nib

    # parameters
    subj_id_file = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    maps_files = '/nfs/m1/hcp/retest/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
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
            data = nib.load(maps_file).get_fdata()
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


def prepare_series(sess=1, run='LR'):
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
    trg_dict = {
        'shape': 'n_subject x n_target x n_time_point',
        'subject': subj_ids,
        'trg_label': trg_labels,
        'rfMRI': np.ones((n_subj, n_trg, n_tp)) * np.nan
    }

    # prepare outputs
    out_seed_lh = pjoin(work_dir, f'rfMRI{sess}_{run}_MPM_lh.pkl')
    out_seed_rh = pjoin(work_dir, f'rfMRI{sess}_{run}_MPM_rh.pkl')
    out_trg = pjoin(work_dir, f'rfMRI{sess}_{run}_MMP.pkl')

    # start
    maps_files = '/nfs/m1/hcp/retest/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
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

    # save
    pkl.dump(hemi2seed_dict['lh'], open(out_seed_lh, 'wb'))
    pkl.dump(hemi2seed_dict['rh'], open(out_seed_rh, 'wb'))
    pkl.dump(trg_dict, open(out_trg, 'wb'))


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
    maps_files = '/nfs/m1/hcp/retest/{0}/MNINonLinear/Results/rfMRI_REST{1}_{2}/'\
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


def rsfc(sess=1, run='LR', hemi='lh'):
    """
    计算静息态功能连接
    """
    import numpy as np
    import pickle as pkl
    from scipy.spatial.distance import cdist

    # prepare seeds
    seed_file = pjoin(work_dir, f'rfMRI{sess}_{run}_ROIv3_{hemi}.pkl')
    seed_dict = pkl.load(open(seed_file, 'rb'))

    # prepare outputs
    out_file = pjoin(work_dir, f'rsfc_ROIv32MMP_{hemi}_{sess}_{run}.pkl')

    # prepare targets
    trg_file = pjoin(work_dir, f'rfMRI{sess}_{run}_MMP.pkl')
    trg_dict = pkl.load(open(trg_file, 'rb'))
    n_trg = len(trg_dict['trg_label'])

    assert seed_dict['subject'] == trg_dict['subject']
    n_subj = len(seed_dict['subject'])

    # prepare FC dictionary
    fc_dict = {
        'shape': 'n_subject x n_target',
        'subject': seed_dict['subject'],
        'trg_label': trg_dict['trg_label']}
    for seed in seed_dict['seed']:
        fc_dict[seed] = np.ones((n_subj, n_trg)) * np.nan

    # start
    for valid_idx in range(n_subj):
        print('Progress: {}/{}'.format(valid_idx+1, n_subj))
        for seed_idx, seed in enumerate(seed_dict['seed']):
            seed_series = seed_dict['rfMRI'][valid_idx, [seed_idx]]
            if not np.isnan(seed_series[0, 0]):
                fc = 1 - cdist(seed_series, trg_dict['rfMRI'][valid_idx],
                               metric='correlation')[0]
                fc_dict[seed][valid_idx] = fc

    # save
    pkl.dump(fc_dict, open(out_file, 'wb'))


def rsfc_mean_among_run(hemi='lh', atlas_name='MPM'):
    import numpy as np
    import pickle as pkl

    # inputs
    sessions = (1, 2)
    runs = ('LR', 'RL')
    fpaths = [pjoin(work_dir, f'rsfc_{atlas_name}2MMP_{hemi}_{ses}_{run}.pkl')
              for ses in sessions for run in runs]
    rois = ['IOG-face', 'pFus-face', 'mFus-face']

    # outputs
    out_file = pjoin(work_dir, f'rsfc_{atlas_name}2MMP_{hemi}.pkl')

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


def rsfc_merge_MMP(hemi='lh', atlas_name='MPM'):
    """
    用ColeAnticevicNetPartition将MMP合并成12个网络
    """
    import numpy as np
    import pickle as pkl
    from scipy.io import loadmat

    # inputs
    seeds = ('IOG-face', 'pFus-face', 'mFus-face')
    rsfc_file = pjoin(work_dir, f'rsfc_{atlas_name}2MMP_{hemi}.pkl')

    # outputs
    out_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_{hemi}.pkl')

    # prepare
    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))
    roi2net_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
        'cortex_parcel_network_assignments.mat'
    roi2net = loadmat(roi2net_file)['netassignments'][:, 0]
    roi2net = np.r_[roi2net[180:], roi2net[:180]]
    net_labels = sorted(set(roi2net))
    n_net = len(net_labels)

    # calculate
    for seed in seeds:
        data = np.zeros((rsfc_dict[seed].shape[0], n_net))
        for net_idx, net_lbl in enumerate(net_labels):
            data[:, net_idx] = np.mean(rsfc_dict[seed][:, roi2net == net_lbl], 1)
        rsfc_dict[seed] = data
    rsfc_dict['trg_label'] = net_labels

    # save
    pkl.dump(rsfc_dict, open(out_file, 'wb'))


def get_rsfc_from_test(hemi='lh', atlas_name='MPM'):
    import pickle as pkl

    # inputs
    new2old = {'MPM': 'mpm', 'ROIv3': 'individual'}
    src_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rfMRI/rsfc_{new2old[atlas_name]}2Cole_{hemi}.pkl')
    subj_file_42 = pjoin(work_dir, 'rfMRI_REST_id')

    # outputs
    out_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_{hemi}_test.pkl')

    # prepare
    subj_ids_42 = open(subj_file_42).read().splitlines()
    data = pkl.load(open(src_file, 'rb'))
    subj_idx_in_data = [data['subject'].index(i) for i in subj_ids_42]

    # calculate
    data['subject'] = subj_ids_42
    for k, v in data.items():
        if k.endswith('-face'):
            data[k] = v[subj_idx_in_data]

    # save
    pkl.dump(data, open(out_file, 'wb'))


def get_roi_idx_vec():
    """
    Get index vector with bool values for each ROI.
    The length of each index vector is matched with 42 subjects.
    True代表该被试在test能标定出各个ROI且也具有4个静息run。
    """
    import numpy as np
    import pandas as pd

    # inputs
    src_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               'rois_v3_idx_vec.csv')
    subj_file_42 = pjoin(work_dir, 'rfMRI_REST_id')
    subj_file_995 = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                          'refined_with_Kevin/rfMRI/rfMRI_REST_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')

    # outputs
    out_file = pjoin(work_dir, 'rois_v3_idx_vec.csv')

    # prepare
    subj_ids_42 = open(subj_file_42).read().splitlines()
    subj_ids_995 = open(subj_file_995).read().splitlines()
    subj_not_995_flag = [i not in subj_ids_995 for i in subj_ids_42]
    print(np.sum(subj_not_995_flag))
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    subj_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_42]
    src_df = pd.read_csv(src_file)

    # calculate
    df = src_df.loc[subj_idx_in_1080]
    df.loc[subj_not_995_flag] = False

    # save
    df.to_csv(out_file, index=False)


def count_roi():
    """
    Count valid subjects for each ROI
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(pjoin(work_dir, 'rois_v3_idx_vec.csv'))
    for col in df.columns:
        print(f'#subjects of {col}:', np.sum(df[col]))


def remove_subjects(hemi='lh', atlas_name='MPM', ses='test'):
    """
    对每个ROI，比如右脑mFus-face，从数据中去掉未标定出该ROI的被试
    """
    import os
    import pandas as pd
    import pickle as pkl

    # inputs
    idx_file = pjoin(work_dir, 'rois_v3_idx_vec.csv')
    if ses == 'test':
        fpath = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_{hemi}_test.pkl')
    elif ses == 'retest':
        fpath = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_{hemi}.pkl')
    else:
        raise ValueError('error session name:', ses)

    # outputs
    fname = os.path.basename(fpath)
    fname = f"{fname.rstrip('.pkl')}_rm-subj.pkl"
    out_file = pjoin(work_dir, fname)

    # prepare
    data = pkl.load(open(fpath, 'rb'))
    idx_df = pd.read_csv(idx_file)

    # calculate
    for k, v in data.items():
        if k == 'subject':
            data[k] = []
        elif k.endswith('-face'):
            hemi_roi = f'{hemi}_{k}'
            data[k] = v[idx_df[hemi_roi]]
            print(hemi_roi, data[k].shape)

    # save
    pkl.dump(data, open(out_file, 'wb'))


def test_retest_icc(atlas_name='MPM'):
    import time
    import numpy as np
    import pickle as pkl
    from cxy_hcp_ffa.lib.heritability import icc

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    test_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_'
                                '{hemi}_test_rm-subj.pkl')
    retest_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_'
                                  '{hemi}_rm-subj.pkl')

    # outputs
    out_file = pjoin(work_dir, f'{atlas_name}_rm-subj_icc.pkl')

    # prepare
    data = {}

    # calculate
    for hemi in hemis:
        rsfc_test = pkl.load(open(test_file.format(hemi=hemi), 'rb'))
        rsfc_retest = pkl.load(open(retest_file.format(hemi=hemi), 'rb'))
        for roi in rois:
            time1 = time.time()
            test_vec = np.mean(rsfc_test[roi], 1)
            retest_vec = np.mean(rsfc_retest[roi], 1)
            k = f"{hemi}_{roi.split('-')[0]}"
            data[k] = icc(test_vec, retest_vec, 10000, 95)
            print(f'Finished {k}: cost {time.time()-time1} seconds.')

    # save
    pkl.dump(data, open(out_file, 'wb'))


def test_retest_corr(atlas_name='MPM'):
    import time
    import pickle as pkl
    from scipy.stats.stats import pearsonr

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    test_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_'
                                '{hemi}_test_rm-subj.pkl')
    retest_file = pjoin(work_dir, f'rsfc_{atlas_name}2Cole_'
                                  '{hemi}_rm-subj.pkl')

    # outputs
    out_file = pjoin(work_dir, f'{atlas_name}_rm-subj_corr.pkl')

    # prepare
    data = {}

    # calculate
    for hemi in hemis:
        rsfc_test = pkl.load(open(test_file.format(hemi=hemi), 'rb'))
        rsfc_retest = pkl.load(open(retest_file.format(hemi=hemi), 'rb'))
        for roi in rois:
            time1 = time.time()
            k = f"{hemi}_{roi.split('-')[0]}"
            v = []
            for vec1, vec2 in zip(rsfc_test[roi], rsfc_retest[roi]):
                v.append(pearsonr(vec1, vec2)[0])
            data[k] = v
            print(f'Finished {k}: cost {time.time()-time1} seconds.')

    # save
    pkl.dump(data, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_valid_id(sess=1, run='LR')
    # get_valid_id(sess=1, run='RL')
    # get_valid_id(sess=2, run='LR')
    # get_valid_id(sess=2, run='RL')
    # get_common_id()
    # prepare_series(sess=1, run='LR')
    # prepare_series(sess=1, run='RL')
    # prepare_series(sess=2, run='LR')
    # prepare_series(sess=2, run='RL')
    # prepare_series_ind(sess=1, run='LR')
    # prepare_series_ind(sess=1, run='RL')
    # prepare_series_ind(sess=2, run='LR')
    # prepare_series_ind(sess=2, run='RL')
    # rsfc(sess=1, run='LR', hemi='lh')
    # rsfc(sess=1, run='LR', hemi='rh')
    # rsfc(sess=1, run='RL', hemi='lh')
    # rsfc(sess=1, run='RL', hemi='rh')
    # rsfc(sess=2, run='LR', hemi='lh')
    # rsfc(sess=2, run='LR', hemi='rh')
    # rsfc(sess=2, run='RL', hemi='lh')
    # rsfc(sess=2, run='RL', hemi='rh')
    # rsfc_mean_among_run(hemi='lh', atlas_name='MPM')
    # rsfc_mean_among_run(hemi='rh', atlas_name='MPM')
    # rsfc_mean_among_run(hemi='lh', atlas_name='ROIv3')
    # rsfc_mean_among_run(hemi='rh', atlas_name='ROIv3')
    # rsfc_merge_MMP(hemi='lh', atlas_name='MPM')
    # rsfc_merge_MMP(hemi='rh', atlas_name='MPM')
    # rsfc_merge_MMP(hemi='lh', atlas_name='ROIv3')
    # rsfc_merge_MMP(hemi='rh', atlas_name='ROIv3')
    # get_rsfc_from_test(hemi='lh', atlas_name='MPM')
    # get_rsfc_from_test(hemi='rh', atlas_name='MPM')
    # get_rsfc_from_test(hemi='lh', atlas_name='ROIv3')
    # get_rsfc_from_test(hemi='rh', atlas_name='ROIv3')
    # get_roi_idx_vec()
    # count_roi()
    remove_subjects(hemi='lh', atlas_name='MPM', ses='test')
    remove_subjects(hemi='rh', atlas_name='MPM', ses='test')
    remove_subjects(hemi='lh', atlas_name='ROIv3', ses='test')
    remove_subjects(hemi='rh', atlas_name='ROIv3', ses='test')
    remove_subjects(hemi='lh', atlas_name='MPM', ses='retest')
    remove_subjects(hemi='rh', atlas_name='MPM', ses='retest')
    remove_subjects(hemi='lh', atlas_name='ROIv3', ses='retest')
    remove_subjects(hemi='rh', atlas_name='ROIv3', ses='retest')
    test_retest_icc(atlas_name='MPM')
    test_retest_icc(atlas_name='ROIv3')
    test_retest_corr(atlas_name='MPM')
    test_retest_corr(atlas_name='ROIv3')
