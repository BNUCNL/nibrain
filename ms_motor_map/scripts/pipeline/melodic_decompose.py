"""
/nfs/m1/BrainImageNet/NaturalObject/data/bold/nifti/sub-core01/ses-ImageNet01/func/
"""

import os, subprocess, re, argparse

class bcolors:
    WARNING = '\033[33m'
    FAIL = '\033[41m'
    BOLD_NODE = '\033[1;32m'
    BOLD = '\033[1;34m'
    ENDC = '\033[0m'

def melodic_decompose(args):

    print(bcolors.BOLD_NODE + "[Node] run MELODIC..." + bcolors.ENDC)

    derivatives_dir = os.path.join(args.projectdir, 'data', 'bold', 'derivatives')

    if args.subject:
        subjects = args.subject
    else:
        subjects = [sub.replace("sub-", "") for sub in os.listdir(os.path.join(derivatives_dir, 'fmriprep')) if "sub-" in sub]

    for subject in subjects:

        if args.session:
            sessions = args.session
        else:
            sessions = [ses.replace("ses-", "") for ses in os.listdir(os.path.join(derivatives_dir, 'fmriprep', 'sub-' + subject)) if "ses-" in ses]

        for session in sessions:

            if args.run:
                runs = args.run
            else:
                runs = []
                for filename in os.listdir(os.path.join(derivatives_dir, 'fmriprep', 'sub-' + subject, session, 'func')):
                    if 'preproc_bold.nii.gz' in filename:
                        runs.append(re.findall('run-(.+?)_space', filename)[0])

            for run in runs:
                func_data = os.path.join(args.fmriprep_output_dir, 'fmriprep', 'sub-' + subject, session, 'func',
                                         'sub-' + subject + '_' + session + '_' + 'task-' + args.taskname + '_' + 'run-' + run + '_space-T1w_desc-preproc_bold.nii.gz')
                ica_output = os.path.join(args.ica_output_dir, 'sub-' + subject, session, 'run-' + run + '.ica')
                melodic_decomposition_command = ' '.join(['melodic',
                                                          '-i', func_data,
                                                          '-o', ica_output,
                                                          '-v',
                                                          '--nobet',
                                                          '--bgthreshold=1',
                                                          '--tr='+args.tr,
                                                          '-d 0',
                                                          '--mmthresh=0.5',
                                                          '--report'])
                print("command: " + bcolors.BOLD + "{:s}".format(melodic_decomposition_command) + bcolors.ENDC)

                if not args.preview:
                    try:
                        subprocess.check_call(melodic_decomposition_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('MELODIC: Error happened in subject {}'.format(subject))

def melodic_integrate(args):
    print('******')


if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("projectdir", help="base dir contains all project files")
    parser.add_argument("taskname", help="name of task")
    parser.add_argument("tr", help="repetition time")

    """
        optinal parameters 
    """
    parser.add_argument("--subject", type=str, nargs="+", help="subjects")
    parser.add_argument("--session", type=str, nargs="+", help="sessions")
    parser.add_argument("--run", type=str, nargs="+", help="runs")
    parser.add_argument("--preview", action="store_true", help="if choose, user can preview the whole pipeline and inspect critical information without runing any process command")
    parser.add_argument("--decompose", action="store_true", help="if choose")
    parser.add_argument("--integrate", action="store_true", help="if choose")

    args = parser.parse_args()

    # melodic_decompose
    if args.decompose:
        melodic_decompose(args)

    if args.integrate:
        melodic_integrate(args)



