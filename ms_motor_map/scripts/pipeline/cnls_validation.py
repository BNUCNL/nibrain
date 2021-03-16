"""
    validation before data2bids
"""

import os, argparse, logging, subprocess
from os.path import join as pjoin

class bcolors:
    WARNING = '\033[33m'
    FAIL = '\033[41m'
    BOLD_NODE = '\033[1;32m'
    BOLD = '\033[1;34m'
    ENDC = '\033[0m'

def log_config(log_name):
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO, filename=log_name, filemode='w')

def runcmd(command, verbose=0, timeout=1200):
    """
    run command line
    """
    ret = subprocess.run(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         encoding="utf-8", timeout=timeout)
    if ret.returncode == 0 and verbose:
        print("success:", ret)
    elif ret.returncode != 0:
        print("error:", ret)
        
def check_path(path, mkdir):
    if type(path) == str:
        if not os.path.exists(path):
            if mkdir:
                os.mkdir(path)
                logging.warning('{} does not exist but validator has been created automatically.'.format(path))
                print('[Warning] {} does not exist but validator has been created automatically.'.format(path))
            else:
                logging.warning('{} does not exist.'.format(path))
                print("[Warning] {} does not exist.".format(path))
        else:
            logging.info('{} already exists.'.format(path))
            print("[info] {} already exists.".format(path))
    else:
        raise AssertionError("Input must be str")

def cnls_validation(args):

    print(bcolors.BOLD_NODE + "[Node] Checking..." + bcolors.ENDC)

    # prepare required path and parameter
    data_dir = pjoin(args.projectdir, 'data')
    exp_dir = pjoin(args.projectdir, 'exp')
    preexp_dir = pjoin(args.projectdir, 'pre_exp')
    
    behavior_dir = pjoin(data_dir, 'behavior')
    bold_dir = pjoin(data_dir, 'bold')
    code_dir = pjoin(data_dir, 'code')
    
    orig_dir = pjoin(bold_dir, 'orig')
    dicom_dir = pjoin(bold_dir, 'dicom')
    nifti_dir = pjoin(bold_dir, 'nifti')
    info_dir = pjoin(bold_dir, 'info')
    derivatives_dir = pjoin(bold_dir, 'derivatives')
    
    work_dir = pjoin(derivatives_dir, 'workdir')
    
    if args.create or args.initialize:
        mkdir = 1
    else:
        mkdir = 0

    # prepare logging
    log_config(pjoin(bold_dir, 'info', 'CNLS_validation.log'))

    # Check if the file/folder exists
    if not args.initialize:
        if not os.path.exists(args.scaninfo):
            logging.critical('scaninfo file in {} is not found!'.format(args.scaninfo))
            raise AssertionError(bcolors.FAIL + "[Critical] NOT FOUND {}".format(args.scaninfo) + bcolors.ENDC)
        if not os.listdir(orig_dir):
            logging.critical('orig_dir in {} is empty!'.format(orig_dir))
            raise AssertionError("[Critical] orig_dir in {} is empty!".format(orig_dir) + bcolors.ENDC)
    for path in [data_dir, exp_dir, preexp_dir, \
                     behavior_dir, bold_dir, code_dir, \
                         orig_dir, dicom_dir, nifti_dir, info_dir,\
                             derivatives_dir, work_dir]:
        check_path(path, mkdir)

    # make soft link to orig dir
    if args.origdir:
        cmd = 'ln -s {} {}'.format(args.origdir, orig_dir)
        runcmd(cmd)

    print('Log is saved in {}.'.format(pjoin(bold_dir, 'info', 'CNLS_validation.log')))

if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("scaninfo", help="path to fetch scaninfo.xlsx", default='data/bold/info/scaninfo.xlsx')
    parser.add_argument("projectdir", help="base dir contains all project files")

    """
        optinal parameters 
    """
    parser.add_argument("--initialize", action="store_true", help="if choose, validator will initialize projectdir.")
    parser.add_argument("--create", action="store_true", help="if choose, validator will create required folder if it is not exist.")
    parser.add_argument("--origdir", help="dir contains original data, if not None, \
                        validator will create a soft link to original data dir. It should an be absolute path.")

    args = parser.parse_args()

    # CNLS validation
    cnls_validation(args)




