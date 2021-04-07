"""
    remove noise ICs
"""

import os, subprocess

# input
projectdir = '/nfs/e4/function_guided_resection/MotorMap'
subject = 'sub-01'
session= 'ses-1'
run= 'run-2'

# first step: read noise ICs number from results.txt
ica_dir = os.path.join(projectdir, 'data', 'bold', 'derivatives', 'melodic', subject, session, run + '.ica')
with open(os.path.join(ica_dir, 'results.txt')) as results:
    noise_id = results.readlines()[-1].replace('[','').replace(']','').replace(' ','')

# second step: run fsl_regfilt
input_file = os.path.join(projectdir, 'data', 'bold', 'derivatives', 'fmriprep', subject, session, 'func', subject + '_' + session + '_' + 'task-motor' + '_' + run + '_space-T1w_desc-preproc_bold.nii.gz')
# output_file = os.path.join(projectdir, 'data', 'bold', 'derivatives', 'fmriprep', subject, session, 'func', subject + '_' + session + '_' + 'task-motor' + '_' + run + '_space-T1w_desc-preproc_bold_denoised.nii.gz')
output_file = '/nfs/s2/userhome/masai/workingdir/melodic_test/s1_ss1_r2_denoised.nii.gz'
mix_file = os.path.join(ica_dir, 'melodic_mix')
fsl_regfilt_command = ' '.join(['fsl_regfilt',
                                '-i', input_file,
                                '-o', output_file,
                                '-d', mix_file,
                                '-f', '"{}"'.format(noise_id)
                                ])
print('RUN CMD: ' + fsl_regfilt_command)
try:
    subprocess.check_call(fsl_regfilt_command, shell=True)
except subprocess.CalledProcessError:
    raise Exception('CIFTIFY_RECON_ALL: Error happened in {}'.format(subject + '_' + session + '_' + run))