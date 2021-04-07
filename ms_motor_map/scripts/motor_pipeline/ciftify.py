"""
    ciftify recon-all & subject-fmri
"""

import subprocess

subject_list = ['sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-16', 'sub-17']

for subject in subject_list:
    ciftify_recon_all_command = ' '.join(['ciftify_recon_all',
                                          '--resample-to-T1w32k',
                                          '--surf-reg', 'MSMSulc',
                                          '--ciftify-work-dir', '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/ciftify',
                                          '--fs-subjects-dir', '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/freesurfer',
                                          subject])
    try:
        subprocess.check_call(ciftify_recon_all_command, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('CIFTIFY_RECON_ALL: Error happened in {}'.format(subject))
