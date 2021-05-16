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
    out_file = pjoin(work_dir, 'rfMRI_REST_4run')

    # calculate
    ids = set(open(fpaths[0]).read().splitlines())
    for f in fpaths[1:]:
        ids.intersection_update(open(f).read().splitlines())
    ids = '\n'.join(sorted(ids))

    # save
    with open(out_file, 'w') as wf:
        wf.write(ids)


if __name__ == '__main__':
    # get_valid_id(sess=1, run='AP')
    # get_valid_id(sess=1, run='PA')
    # get_valid_id(sess=2, run='AP')
    # get_valid_id(sess=2, run='PA')
    get_common_id()
