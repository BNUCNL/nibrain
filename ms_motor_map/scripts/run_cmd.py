"""
    run cmd
"""

import os, argparse, subprocess, re

def run_cmd(args):
    # input
    cmd = args.cmd
    projectdir = args.projectdir

    # subject
    if args.subject:
        subjects = args.subject
    else:
        subjects = [sub.replace("sub-", "") for sub in os.listdir(os.path.join(projectdir, 'data', 'bold', 'nifti')) if "sub-" in sub]
    for subject in subjects:
        # session
        if args.session:
            sessions = args.session
        else:
            sessions = [ses.replace("ses-", "") for ses in os.listdir(os.path.join(projectdir, 'data', 'bold', 'nifti', 'sub-' + subject)) if "ses-" in ses]
        for session in sessions:
            # run
            if args.run:
                runs = args.run
            else:
                runs = []
                for filename in os.listdir(os.path.join(projectdir, 'data', 'bold', 'nifti', 'sub-' + subject, 'ses-' + session, 'func')):
                    if '_bold.nii.gz' in filename:
                        runs.append(re.findall('run-(.+?)_bold', filename)[0])
            for run in runs:
                # adjust cmd
                if '<subject>' in cmd:
                    cmd1 = cmd.replace('<subject>', subject)
                if '<session>' in cmd:
                    cmd1 = cmd1.replace('<session>', session)
                if '<run>' in cmd:
                    cmd1 = cmd1.replace('<run>', str(int(run)))
                print(cmd1)
                # run cmd
                if not args.preview:
                    try:
                        subprocess.check_call(cmd, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('RUN CMD: Error happened in subject {}'.format(subject))

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
    parser.add_argument('-c', '--cmd', help="command, use double quotation")
    parser.add_argument('-s', "--subject", type=str, nargs="+", help="subject id")
    parser.add_argument('-ss', "--session", type=str, nargs="+", help="session id")
    parser.add_argument('-r', "--run", type=str, nargs="+", help="run id")
    parser.add_argument('-p', "--preview", action="store_true", help="command line preview")

    args = parser.parse_args()

    # run cmd
    run_cmd(args)