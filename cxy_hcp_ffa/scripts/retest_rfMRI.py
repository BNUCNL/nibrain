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
    prepare_series_ind(sess=1, run='LR')
    prepare_series_ind(sess=1, run='RL')
    prepare_series_ind(sess=2, run='LR')
    prepare_series_ind(sess=2, run='RL')
