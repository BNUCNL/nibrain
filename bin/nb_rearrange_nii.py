#! /usr/local/neurosoft/miniconda3/bin/python
"""
Rearrange nifti files created by dicom2nifti.py to BIDS style.

Edited by Xiayu Chen, 2019-04-16
Last modified by Xiayu Chen, 2019-04-27
"""

import os
import time
import glob
import json
import logging
import subprocess
import nibabel as nib

from os.path import join as pjoin

t1_protocol_name = 't1_mprage_sag'
rest_protocol_name = 'ge_func_3x3x3p5_240_RS'
task_protocol_name = 'ge_func_pace_3x3x4_99'

log_dir = '/nfs/h2/development/migration_BIDS/log_files'
local_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    filename=pjoin(log_dir, 'rearrange_nii_log_{}'.format(local_time)),
    filemode='w'
)
logger = logging.getLogger()

for tmp_dir in glob.iglob('/nfs/h2/development/migration_BIDS/sub-*/*/tmp'):
    sess_dir = os.path.dirname(tmp_dir)
    subj_dir, sess = os.path.split(sess_dir)
    subj = os.path.basename(subj_dir)
    tmp_nii_dir = pjoin(tmp_dir, 'nii')
    tmp_run_dir = pjoin(tmp_dir, 'run_info')

    rearranged_items = []
    for tmp_file in os.listdir(tmp_nii_dir):
        # check file
        item = tmp_file.split('.')[0]
        if item in rearranged_items:
            continue
        nii_file = pjoin(tmp_nii_dir, item+'.nii.gz')
        json_file = pjoin(tmp_nii_dir, item+'.json')
        if not os.path.exists(nii_file) or not os.path.exists(json_file):
            msg = '{} is lack of a pair of JSON and NIFTI files'.format(pjoin(tmp_nii_dir, tmp_file))
            print(msg)
            logger.warning(msg)
            continue

        # rearrange file
        json_data = json.load(open(json_file))
        if json_data['ProtocolName'] == t1_protocol_name:
            rlf = pjoin(tmp_run_dir, 'mri.rlf')
            runs = open(rlf).read().splitlines()
            if str(json_data['SeriesNumber']) not in runs:
                continue
            anat_dir = pjoin(sess_dir, 'anat')
            if not os.path.exists(anat_dir):
                os.makedirs(anat_dir)
            item_new = '_'.join([subj, sess, 'run-{}'.format(json_data['SeriesNumber']), 'T1w'])
            nii_file_new = pjoin(anat_dir, item_new+'.nii.gz')
            json_file_new = pjoin(anat_dir, item_new+'.json')
        elif json_data['ProtocolName'] == rest_protocol_name:
            rlf = pjoin(tmp_run_dir, 'rfmri.rlf')
            runs = open(rlf).read().splitlines()
            if str(json_data['SeriesNumber']) not in runs:
                continue
            func_dir = pjoin(sess_dir, 'func')
            if not os.path.exists(func_dir):
                os.makedirs(func_dir)
            item_new = '_'.join([subj, sess, 'task-rest', 'run-{}'.format(json_data['SeriesNumber']), 'bold'])
            nii_file_new = pjoin(func_dir, item_new+'.nii.gz')
            json_file_new = pjoin(func_dir, item_new+'.json')
        elif json_data['ProtocolName'] == task_protocol_name:
            rlf = pjoin(tmp_run_dir, 'obj.rlf')
            runs = open(rlf).read().splitlines()
            if str(json_data['SeriesNumber']) not in runs:
                continue
            func_dir = pjoin(sess_dir, 'func')
            if not os.path.exists(func_dir):
                os.makedirs(func_dir)
            item_new = '_'.join([subj, sess, 'task-obj', 'run-{}'.format(json_data['SeriesNumber']), 'bold'])
            nii_file_new = pjoin(func_dir, item_new+'.nii.gz')
            json_file_new = pjoin(func_dir, item_new+'.json')
        else:
            continue
        mv_cmd1 = 'mv -v {} {}'.format(nii_file, nii_file_new)
        subprocess.call(mv_cmd1, shell=True)
        logger.info(mv_cmd1)
        mv_cmd2 = 'mv -v {} {}'.format(json_file, json_file_new)
        subprocess.call(mv_cmd2, shell=True)
        logger.info(mv_cmd2)
        rearranged_items.append(item)

