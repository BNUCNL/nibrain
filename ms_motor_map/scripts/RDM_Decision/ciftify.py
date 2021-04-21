"""
    ciftify recon-all & subject-fmri
"""

import subprocess, os

subject_list = ['sub-h27']
run_list = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6']

# recon all
for subject in subject_list:
    ciftify_recon_all_command = ' '.join(['ciftify_recon_all',
                                          '--resample-to-T1w32k',
                                          '--surf-reg', 'MSMSulc',
                                          '--ciftify-work-dir', '/nfs/e1/RandomDecision/data/bold/derivatives/ciftify',
                                          '--fs-subjects-dir', '/nfs/e1/RandomDecision/data/bold/derivatives/freesurfer',
                                          subject])
    try:
        subprocess.check_call(ciftify_recon_all_command, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('CIFTIFY_RECON_ALL: Error happened in {}'.format(subject))

    # subject_fmri sub-h27_task-rdm_run-1_space-T1w_desc-preproc_bold.nii.gz
    for run in run_list:
        input_func = os.path.join('/nfs/e1/RandomDecision/data/bold/derivatives/fmriprep/sub-h27/func/', subject + '_task-rdm_' + run + '_space-T1w_desc-preproc_bold.nii.gz')
        ciftify_subfmri_command = ' '.join(['ciftify_subject_fmri',
                                            '--ciftify-work-dir', '/nfs/e1/RandomDecision/data/bold/derivatives/ciftify',
                                            '--surf-reg', 'MSMSulc',
                                            input_func,
                                            subject,
                                            'task-rdm_' + run])
        try:
            subprocess.check_call(ciftify_subfmri_command, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('CIFTIFY_SUBFMRI: Error happened in {}'.format(subject))

        print('finish')


