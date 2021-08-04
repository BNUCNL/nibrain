#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:00:43 2021

@author: masai
"""

import os
from gbc_alff import CapAtlas, global_brain_conn, alff, Ciftiwrite
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import nibabel as nib

data_dir = '/nfs/e1/HCPD/fmriresults01/'
work_dir = '/nfs/e1/HCPD_CB/mri/'

cap = CapAtlas('/nfs/e2/workingshop/masai/code/cb/atlas/ColeAnticevicNetPartition/')

# subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None)[0].tolist()
# for subject in subject_list:
    # for run in ['1_AP', '1_PA', '2_AP', '2_PA']:
subject = 'HCD2063034_V1_MR'
run = '1_AP'

resting_ts = nib.load(os.path.join(data_dir, subject, 'MNINonLinear', 'Results', 'rfMRI_REST'+run, 'rfMRI_REST'+run+'_Atlas_MSMAll_hp0_clean.dtseries.nii'))
cerebellum_LR = cap.get_cerebellum(hemisphere='LR').any(axis=0)

# ALFF & fALFF
alff_falff_cb = alff(cifti_ts=resting_ts, src_roi=cerebellum_LR, tr=0.8, low_freq_band=(0.008, 0.1))
alff_save_dir = os.path.join(work_dir, 'func', 'rest', 'ALFF', 'rfMRI_REST'+run)
if not os.path.exist(alff_save_dir):
    os.makedirs(alff_save_dir)
Ciftiwrite(file_path=os.path.join(alff_save_dir, 'rfMRI_REST'+run+'_alff_falff.dtseries.nii'), data=alff_falff_cb, cifti_ts=resting_ts, src_roi=cerebellum_LR)

# GBC
# cortex
cortex_L = cap.get_cortex(hemisphere='L')
cerebellum_cortex_L_conn = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=cortex_L)
cortex_R = cap.get_cortex(hemisphere='R')
cerebellum_cortex_R_conn = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=cortex_R)
# subcortex
subcortex_L = cap.get_subcortex(hemisphere='L')
cerebellum_subcortex_L_conn = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=cortex_L)
subcortex_R = cap.get_subcortex(hemisphere='R')
cerebellum_subcortex_R_conn = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=cortex_R)
# CAP networks
network_L = cap.get_network(hemisphere='L')
cerebellum_network_L_conn = np.zeros((17853, 12))
for network in np.arange(12):
    cerebellum_network_L_conn[:, network] = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=network_L[network, :])
network_R = cap.get_network(hemisphere='R')
cerebellum_network_R_conn = np.zeros((17853, 12))
for network in np.arange(12):
    cerebellum_network_R_conn[:, network] = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=network_R[network, :])
# CAP parcels
parcel_LR = cap.get_parcel()
cerebellum_parcel_LR_conn = np.zeros((17853, 718))
for parcel in np.arange(718):
    cerebellum_parcel_LR_conn[:, parcel] = global_brain_conn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=parcel_LR[parcel, :])

# save gbc
gbc_save_dir = os.path.join(work_dir, 'func', 'rest', 'GBC', 'rfMRI_REST'+run)
if not os.path.exist(gbc_save_dir):
    os.makedirs(gbc_save_dir)
# left hemisphere
cerebellum_cortex_subcortex_network_L_conn = np.concatenate((cerebellum_cortex_L_conn, cerebellum_subcortex_L_conn, cerebellum_network_L_conn), axis=1)
Ciftiwrite(file_path=os.path.join(gbc_save_dir, 'rfMRI_REST'+run+'_cerebellum_cortex_subcortex_network_L_conn.dtseries.nii'),
           data=cerebellum_cortex_subcortex_network_L_conn, cifti_ts=resting_ts, src_roi=cerebellum_LR)
# right hemisphere
cerebellum_cortex_subcortex_network_R_conn = np.concatenate((cerebellum_cortex_R_conn, cerebellum_subcortex_R_conn, cerebellum_network_R_conn), axis=1)
Ciftiwrite(file_path=os.path.join(gbc_save_dir, 'rfMRI_REST'+run+'_cerebellum_cortex_subcortex_network_R_conn.dtseries.nii'),
           data=cerebellum_cortex_subcortex_network_R_conn, cifti_ts=resting_ts, src_roi=cerebellum_LR)
# 718 parcels
Ciftiwrite(file_path=os.path.join(gbc_save_dir, 'rfMRI_REST'+run+'_cerebellum_parcel_conn.dtseries.nii'),
           data=cerebellum_parcel_LR_conn, cifti_ts=resting_ts, src_roi=cerebellum_LR)