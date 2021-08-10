#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:00:43 2021

@author: masai
"""

import os
from gbc_alff import CapAtlas, compute_fconn, compute_gbc, alff, Ciftiwrite
import numpy as np
import pandas as pd
import nibabel as nib

# data_dir: directory of HCPD data
data_dir = '/nfs/e1/HCPD/fmriresults01/'
# work_dir: directory to save results
work_dir = '/nfs/e1/HCPD_CB/mri/'
# define a class of ColeAnticevicNetPartition to get ROI masks
cap = CapAtlas('/nfs/e2/workingshop/masai/code/cb/atlas/ColeAnticevicNetPartition/')
# get all HCPD subject id from subject_list.csv
subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None)[0].tolist()

for subject in subject_list:

    for run in ['1_AP', '1_PA', '2_AP', '2_PA']:

        # read resting state timeseries file
        resting_ts = nib.load(os.path.join(data_dir, subject, 'MNINonLinear', 'Results', 'rfMRI_REST' + run,
                                           'rfMRI_REST' + run + '_Atlas_MSMAll_hp0_clean.dtseries.nii'))
        # left&right cerebellum mask
        cerebellum_LR = cap.get_cerebellum(hemisphere='LR').any(axis=0)

        # ALFF & fALFF
        alff_falff_cb = alff(cifti_ts=resting_ts, src_roi=cerebellum_LR, tr=0.8, low_freq_band=(0.008, 0.1))
        # save
        alff_save_dir = os.path.join(work_dir, subject, 'func', 'rest', 'ALFF', 'rfMRI_REST' + run)
        if not os.path.exists(alff_save_dir):
            os.makedirs(alff_save_dir)
        Ciftiwrite(file_path=os.path.join(alff_save_dir, 'rfMRI_REST' + run + '_alff_falff.dtseries.nii'),
                   data=alff_falff_cb, cifti_ts=resting_ts, src_roi=cerebellum_LR)

        # GBC
        # compute functional connectivitry
        conn = compute_fconn(cifti_ts=resting_ts, src_roi=cerebellum_LR, targ_roi=None)
        # target roi
        # left&right cortex mask
        cortex_L = cap.get_cortex(hemisphere='L')
        cortex_R = cap.get_cortex(hemisphere='R')
        # left&right subcortex mask
        subcortex_L = cap.get_subcortex(hemisphere='L')
        subcortex_R = cap.get_subcortex(hemisphere='R')
        # left&right networks mask
        network_L = cap.get_network(hemisphere='L')
        network_R = cap.get_network(hemisphere='L')
        for network in np.arange(network_L.shape[0]):
            network_L[network, :] = ~network_L[network, :] & cerebellum_LR
            network_R[network, :] = ~network_R[network, :] & cerebellum_LR
        # left&right parcels mask
        # parcels left ROI-id list
        parcel_L = cap.get_parcel(roi_list=cap.annot[cap.annot['LABEL'].str.contains('L-')]['KEYVALUE'].tolist())
        for parcel in np.arange(parcel_L.shape[0]):
            parcel_L[parcel, :] = ~parcel_L[parcel, :] & cerebellum_LR
        # parcels right ROI-id list
        parcel_R = cap.get_parcel(roi_list=cap.annot[cap.annot['LABEL'].str.contains('R-')]['KEYVALUE'].tolist())
        for parcel in np.arange(parcel_R.shape[0]):
            parcel_R[parcel, :] = ~parcel_R[parcel, :] & cerebellum_LR
        # combine left target ROI & right target ROI
        target_L = np.concatenate((cortex_L.reshape(1, -1), subcortex_L.reshape(1, -1), network_L, parcel_L), axis=0)
        target_R = np.concatenate((cortex_R.reshape(1, -1), subcortex_R.reshape(1, -1), network_R, parcel_R), axis=0)
        # compute GBC
        gbc_L = compute_gbc(conn=conn, targ_group=target_L)
        gbc_R = compute_gbc(conn=conn, targ_group=target_R)
        # save
        gbc_save_dir = os.path.join(work_dir, 'func', 'rest', 'GBC', 'rfMRI_REST' + run)
        if not os.path.exists(gbc_save_dir):
            os.makedirs(gbc_save_dir)
        # save left hemisphere
        Ciftiwrite(file_path=os.path.join(gbc_save_dir,
                                          'rfMRI_REST' + run + '_cerebellum_conn_L.dtseries.nii'),
                   data=gbc_L, cifti_ts=resting_ts, src_roi=cerebellum_LR)
        # save right hemisphere
        Ciftiwrite(file_path=os.path.join(gbc_save_dir,
                                          'rfMRI_REST' + run + '_cerebellum_conn_R.dtseries.nii'),
                   data=gbc_R, cifti_ts=resting_ts, src_roi=cerebellum_LR)
