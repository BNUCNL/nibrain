"""
    batch processing of melody command line
"""



import os
import subprocess

def melodic(fmriprep_dir, ica_output_dir, subject_id, session_id, run_id):
    func_data = os.path.join(fmriprep_dir, subject_id, session_id, 'func',
                             subject_id + '_' + session_id + '_' + 'task-motor' + '_' + run_id + '_space-T1w_desc-preproc_bold.nii.gz')
    ica_output = os.path.join(ica_output_dir, subject_id, session_id, run_id + '.ica')
    melodic_command = ' '.join(['melodic', '-i', func_data, '-o', ica_output,
                                '-v --nobet --bgthreshold=1 --tr=2 -d 0 --mmthresh=0.5 --report --guireport=../../report.html'])
    try:
        subprocess.check_call(melodic_command, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('MELODIC: Error happened in subject {}'.format(subject_id))

if __name__ == '__main__':
    fmriprep_dir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/fmriprep/fmriprep'
    ica_output_dir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/melodic'
    raw_data_dir = '/nfs/e4/function_guided_resection/MotorMapping'

    subject_list = []
    all_subject_folders = os.listdir(fmriprep_dir)
    for foldername in all_subject_folders:
        if 'sub-M' in foldername and 'html' not in foldername:
            subject_list.append(foldername)

    for subject_id in subject_list:
        subject_dir = os.path.join(ica_output_dir, subject_id)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        session_id = 'ses-01'
        session_dir = os.path.join(subject_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        with open(os.path.join(raw_data_dir, subject_id, session_id, 'tmp', 'run_info', 'motor.rlf'), 'r') as f:
            run_list = f.read().splitlines()
        for run_id in run_list:
            run_id = 'run-' + run_id
            run_dir = os.path.join(session_dir, run_id + '.ica')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            melodic(fmriprep_dir, ica_output_dir, subject_id, session_id, run_id)
            print(subject_id + '-' + session_id + '-' + run_id + ' completed!')