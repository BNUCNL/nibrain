#! /usr/local/neurosoft/miniconda3/bin/python
"""
Reconstruct cortical surface for sessions own T1w data

Edited by Xiayu Chen, 2019-04-27
Last modified by Xiayu Chen, 2019-04-27
"""

import os
import time
import glob
import logging
import subprocess

from os.path import join as pjoin

data_dir = '/nfs/h2/development/migration_BIDS'
sub_ses_id_file = pjoin(data_dir, 'sub_ses_id_4')
trg_dir = pjoin(data_dir, 'derivatives/corticalsurface')
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

sub_ses_ids = open(sub_ses_id_file).read().splitlines()

local_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    filename=pjoin(data_dir, 'log_files/recon_all-sess_log_{}'.format(local_time)),
    filemode='w'
)
logger = logging.getLogger()

for sub_ses_id in sub_ses_ids:
    t1_files = glob.glob(pjoin(data_dir, sub_ses_id, 'anat/*T1w.nii.gz'))
    if not t1_files:
        continue
    sessid = sub_ses_id.split('-')[-1]
    recon_all_cmd = 'recon-all -subjid ' + sessid
    for t1_file in t1_files:
        recon_all_cmd = recon_all_cmd + ' -i ' + t1_file
    recon_all_cmd = recon_all_cmd + ' -all -sd ' + trg_dir
    subprocess.call(recon_all_cmd, shell=True)
    logger.info(recon_all_cmd)

