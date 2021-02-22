from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/rfMRI')


def get_valid_id(sess=1, run='LR'):
    import os
    import time
    from commontool.io.io import CiftiReader

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


def fc_individual(hemi='lh', sess=1, run='LR'):
    import numpy as np
    import pickle as pkl
    from scipy.spatial.distance import cdist

    # parameters
    seed_file = pjoin(work_dir, f'rfMRI_seed_{hemi}_{sess}_{run}.pkl')
    trg_file = pjoin(work_dir, f'rfMRI_trg_{sess}_{run}.pkl')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')

    # outputs
    out_file = pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_{sess}_{run}.pkl')

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
                fc = 1 - cdist(seed_series, trg_dict['rfMRI'][valid_idx], metric='correlation')[0]
                fc_dict[seed][subj_idx] = fc

    pkl.dump(fc_dict, open(out_file, 'wb'))


def fc_mean_among_run(hemi='lh'):
    import numpy as np
    import pickle as pkl

    # parameters
    files = [
        pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_1_LR.pkl'),
        pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_1_RL.pkl'),
        pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_2_LR.pkl'),
        pjoin(work_dir, f'rsfc_individual2MMP_{hemi}_2_RL.pkl'),
    ]
    rois = ['IOG-face', 'pFus-face', 'mFus-face']

    # outputs
    out_file = pjoin(work_dir, f'rsfc_individual2MMP_{hemi}.pkl')

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


if __name__ == '__main__':
    # get_valid_id(1, 'LR')
    # get_valid_id(1, 'RL')
    # get_valid_id(2, 'LR')
    # get_valid_id(2, 'RL')
    # get_common_id()
    fc_individual(hemi='lh', sess=1, run='LR')
    fc_individual(hemi='lh', sess=1, run='RL')
    fc_individual(hemi='lh', sess=2, run='LR')
    fc_individual(hemi='lh', sess=2, run='RL')
    fc_individual(hemi='rh', sess=1, run='LR')
    fc_individual(hemi='rh', sess=1, run='RL')
    fc_individual(hemi='rh', sess=2, run='LR')
    fc_individual(hemi='rh', sess=2, run='RL')
    fc_mean_among_run(hemi='lh')
    fc_mean_among_run(hemi='rh')
