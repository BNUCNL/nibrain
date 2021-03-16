"""
fmriprep-docker
/nfs/e4/function_guided_resection/MotorMapping
/nfs/e4/function_guided_resection/MotorMapping/derivatives/fmriprep_test
participant
-w /nfs/e4/function_guided_resection/fmriprep_tmp_ms
--participant_label M01
--output-space T1w
--skip-bids-validation
--fs-license-file /usr/local/neurosoft/freesurfer/license.txt
"""

import subprocess, argparse, os


class bcolors:
    WARNING = '\033[33m'
    FAIL = '\033[41m'
    BOLD_NODE = '\033[1;32m'
    BOLD = '\033[1;34m'
    ENDC = '\033[0m'

def run_fmriprep(args):

    print(bcolors.BOLD_NODE + "[Node] run FMRIPREP..." + bcolors.ENDC)

    nifti_dir = os.path.join(args.projectdir, 'data', 'bold', 'nifti')
    derivatives_dir = os.path.join(args.projectdir, 'data', 'bold', 'derivatives')

    if args.subject:
        subjects = args.subject
    else:
        subjects = [_.replace("sub-", "") for _ in os.listdir(nifti_dir) if "sub-" in _]

    for subject in subjects:
        fmriprep_command = ' '.join(['fmriprep-docker',
                                   nifti_dir,
                                   derivatives_dir,
                                   'participant',
                                   '-w', args.workdir,
                                   '--participant_label', subject,
                                   '--output-space', 'T1w fsnative',
                                   '--skip-bids-validation',
                                   '--fs-license-file', '/usr/local/neurosoft/freesurfer/license.txt'])
        print("command: " + bcolors.BOLD + "{:s}".format(fmriprep_command) + bcolors.ENDC)
        if not args.preview:
            try:
                subprocess.check_call(fmriprep_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('FMRIPREP: Error happened in subject {}'.format(args.subject))

if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("projectdir", help="base dir contains all project files")
    parser.add_argument("workdir", help="name of directory stores temp files")

    """
        optinal parameters 
    """
    # parser.add_argument("--create", action="store_true", help="if choose, script will create required folder if it is not exist.")
    parser.add_argument("--subject", type=str, nargs="+", help="subjects")
    parser.add_argument("--preview", action="store_true", help="if choose, user can preview the whole pipeline and inspect critical information without runing any process command")
    # parser.add_argument("--options", type=str, nargs="+", help="Add parameters to the command line, There is no need to add '--' before the option")

    args = parser.parse_args()

    # CNLS validation
    run_fmriprep(args)