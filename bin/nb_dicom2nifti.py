#! /usr/local/neurosoft/miniconda3/bin/python
"""
Unpack dicom files to BIDS style.

Edited by Xiayu Chen, 2019-04-16
Last modified by Xiayu Chen, 2019-04-26
"""

import os
import time
import logging
import subprocess

from os.path import join as pjoin

# prepare path
data_dir = '/nfs/h2/development/migration_BIDS'
dicom_par = pjoin(data_dir, 'sourcedata/dicom')
dicom_id_file = pjoin(dicom_par, 'dicomid_M')
scanlist_file = pjoin(dicom_par, 'scanList_M.csv')

# prepare information
dicom_ids = open(dicom_id_file).read().splitlines()
scanlist = [line.split(',') for line in open(scanlist_file).read().splitlines()]
run_types = ['rfmri', 'obj', 'mri']
scandict = dict()
for col in zip(*scanlist[7:]):
    if col[0] in run_types:
        if col[0] in scandict.keys():
            scandict[col[0]].append(col[1:])
        else:
            scandict[col[0]] = [col[1:]]
    else:
        scandict[col[0]] = col[1:]

# prepare logging
local_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    filename=os.path.join(data_dir, 'log_files/dicom2nifti_log_{}'.format(local_time)),
    filemode='w'
)
logger = logging.getLogger()

# start unpacking
for dicom_id in dicom_ids:
    if dicom_id in scandict['dicomid']:
        idx = scandict['dicomid'].index(dicom_id)
        subj_dir = pjoin(data_dir, 'sub-'+scandict['subjid'][idx])
        sess_dir = pjoin(subj_dir, 'ses-'+scandict['sessid'][idx])
        tmp_dir = pjoin(sess_dir, 'tmp')
        nii_dir = pjoin(tmp_dir, 'nii')
        run_dir = pjoin(tmp_dir, 'run_info')
        mkdir_cmd1 = 'mkdir -p ' + nii_dir
        mkdir_cmd2 = 'mkdir ' + run_dir
        subprocess.call(mkdir_cmd1, shell=True)
        subprocess.call(mkdir_cmd2, shell=True)

        # dicom to nifti
        dicom_dir = pjoin(dicom_par, dicom_id)
        dcm2nii_cmd = 'dcm2niix -o ' + nii_dir + ' -z y ' + dicom_dir
        logger.info(dcm2nii_cmd)
        subprocess.call(dcm2nii_cmd, shell=True)

        # run information
        for run_type in run_types:
            runs = set([col[idx] for col in scandict[run_type]])
            runs.discard('')
            runs = sorted(runs, key=lambda x: int(x))
            runs_out = '\n'.join(runs)
            rlf = pjoin(run_dir, run_type+'.rlf')
            open(rlf, 'w+').writelines(runs_out)
    else:
        warning_info = 'dicom ID - {} is not in scanList file!'.format(dicom_id)
        print(warning_info)
        logger.warning(warning_info)

