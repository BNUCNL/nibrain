"""
    after hand classification of ICs
    fsl_regfilt -i /nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/fmriprep/sub-01/ses-1/func/sub-01_ses-1_task-motor_run-2_space-T1w_desc-preproc_bold.nii.gz
    -o /nfs/s2/userhome/masai/workingdir/melodic_test/s1_ss1_r2_denoised.nii.gz
    -d /nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/melodic/sub-01/ses-1/run-2.ica/melodic_mix
    -f "id"
"""

import os, subprocess, argparse

def run_regfilt(args):
    project_dir = args.projectdir
    subject = str(args.subject[0])
    session = str(args.session[0])
    run = str(args.run[0])
    taskname = args.taskname
    input_bold = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'fmriprep', 'sub-' + subject, 'ses-' + session, 'func', 'sub-' + subject + '_' + 'ses-' + session + '_' + 'task-' + taskname + '_' + 'run-' + run + '_space-T1w_desc-preproc_bold.nii.gz')
    output_bold = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'fmriprep', 'sub-' + subject, 'ses-' + session, 'func', 'sub-' + subject + '_' + 'ses-' + session + '_' + 'task-' + taskname + '_' + 'run-' + run + '_space-T1w_desc-preproc_bold_denoised.nii.gz')
    mix_file = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'melodic', 'sub-' + subject, 'ses-' + session, 'run-' + run + '.ica', 'melodic_mix')
    results_file = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'melodic', 'sub-' + subject, 'ses-' + session, 'run-' + run + '.ica', 'results')
    # read noise-ICs id
    with open(results_file) as f:
        ic_id = f.readlines()[-1].replace('[', '').replace(']', '')

    regfilt_cmd = ' '.join(['fsl_regfilt', '-i', input_bold, '-o', output_bold, '-d', mix_file, '-f', '"' + ic_id + '"'])
    print('REGFILT CMD: ' + regfilt_cmd)
    if not args.preview:
        try:
            subprocess.check_call(regfilt_cmd, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('REGFILT CMD: Error happened in subject {}'.format(subject))

if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("projectdir", help="projectdir")

    """
        optinal parameters 
    """
    parser.add_argument('-s', "--subject", type=str, nargs="+", help="subject id")
    parser.add_argument('-ss', "--session", type=str, nargs="+", help="session id")
    parser.add_argument('-r', "--run", type=str, nargs="+", help="run id")
    parser.add_argument('-tn', "--taskname", help="task name")
    parser.add_argument('-p', "--preview", action="store_true", help="command line preview")

    args = parser.parse_args()

    # run cmd
    run_regfilt(args)