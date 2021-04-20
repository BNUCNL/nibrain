"""
    preprocessing_fsaverage
    /nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/
    /nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/fmriprep/sub-01/ses-1/func/
    preproc-sess -sf Sesslist -fsd "unsmth_original" -surface fsaverage lhrh -fwhm 0 -per-run -nostc -noinorm -nomc
    ！！！不能用！！！
"""

import os, subprocess

subject_list = ['sub-01']
run_list = ['run-2']
project_dir = '/nfs/e4/function_guided_resection/MotorMap'

denoise_validation_dir = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'denoise_validation')
for subject_id in subject_list:
    for run_id in run_list:

        # # create FS Directory Structure
        # if not os.path.exists(os.path.join(denoise_validation_dir, 'before_data', subject_id.replace('sub-', 'Sess'), 'bold', run_id.replace('run-', '00'))):
        #     os.makedirs(os.path.join(denoise_validation_dir, 'before_data', subject_id.replace('sub-', 'Sess'), 'bold', run_id.replace('run-', '00')))
        # if not os.path.exists(os.path.join(denoise_validation_dir, 'after_data', subject_id.replace('sub-', 'Sess'), 'bold', run_id.replace('run-', '00'))):
        #     os.makedirs(os.path.join(denoise_validation_dir, 'after_data', subject_id.replace('sub-', 'Sess'), 'bold', run_id.replace('run-', '00')))
        #
        # # link func data to dir
        # before_data_path = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'fmriprep', subject_id, 'ses-1', 'func', '{0}_ses-1_task-motor_{1}_space-T1w_desc-preproc_bold.nii.gz'.format(subject_id, run_id))
        # if not os.path.exists(before_data_path):
        #     before_ln_cmd = ' '.join(['ln', '-s', before_data_path, os.path.join(denoise_validation_dir, 'before_data', subject_id.replace('sub-', 'Sess'), 'bold', run_id.replace('run-', '00'), '{0}_ses-1_task-motor_{1}_space-T1w_desc-preproc_bold.nii.gz'.format(subject_id, run_id))])
        #     try:
        #         subprocess.check_call(before_ln_cmd, shell=True)
        #     except subprocess.CalledProcessError:
        #         raise Exception('CMD: Error happened in subject {}'.format(subject_id))
        # # after_data_path = ''

        # Register fMRI volume data on 'fsaverage' surface template using FreeSurfer version 6.0.0
        # before
        os.chdir(os.path.join(denoise_validation_dir, 'before_data'))
        before_reg_cmd = ' '.join(['preproc-sess', '-s', 'sub-01', '-fsd', 'bold', '-surface fsaverage lhrh -fwhm 0 -per-run -nostc -noinorm -nomc'])
        print(before_reg_cmd)
        try:
            subprocess.check_call(before_reg_cmd, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('REG: Error happened in subject {}'.format(subject_id))