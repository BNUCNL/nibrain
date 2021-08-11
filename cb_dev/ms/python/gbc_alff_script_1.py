#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:00:43 2021

@author: masai
"""

import os
from gbc_alff import CapAtlas, compute_fconn, compute_gbc, alff, Ciftiwrite
from ioTools import CiftiReader, save2nifti
import numpy as np
import pandas as pd
import nibabel as nib

# data_dir: directory of HCPD data
data_dir = '/nfs/e1/HCPD/fmriresults01/'
# work_dir: directory to save results
work_dir = '/nfs/e1/HCPD_CB/mri/'
# get all HCPD subject id from subject_list.csv
subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None)[0].tolist()
# define a class of ColeAnticevicNetPartition to get ROI masks
cap = CapAtlas('/nfs/e2/workingshop/masai/code/cb/atlas/ColeAnticevicNetPartition/')

# left&right cerebellum mask
cerebellum_LR = cap.get_cerebellum(hemisphere='LR').any(axis=0)
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

# cerebellum volume index & affine
resting_img = CiftiReader('/nfs/e1/HCPD/fmriresults01/HCD0008117_V1_MR/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii')
affine_matrix = resting_img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
cb_L_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT', zeroize=False)[2]
cb_R_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_RIGHT', zeroize=False)[2]
cb_L_index.extend(cb_R_index)
cb_index_matrix = np.asarray(cb_L_index)

# for subject in subject_list:
for subject in ['HCD0001305_V1_MR']:

    #
    save_dir = os.path.join(work_dir, subject, 'MNINonLinear', 'Results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #
    # alff_falff = np.zeros((cerebellum_LR.shape[0], 8))
    # fc = np.zeros((cerebellum_LR.shape[0], 91282*4))
    alff_falff = list()
    fc = list()

    for index, run in enumerate(['1_AP', '1_PA', '2_AP', '2_PA']):

        # read resting state timeseries file
        resting_ts = nib.load(os.path.join(data_dir, subject, 'MNINonLinear', 'Results', 'rfMRI_REST' + run,
                                           'rfMRI_REST' + run + '_Atlas_MSMAll_hp0_clean.dtseries.nii'))
        # ALFF & fALFF
        alff_falff_cb = alff(resting_ts, cerebellum_LR, tr=0.8, low_freq_band=(0.008, 0.1))
        #
        # alff_falff[:, (2*index):(2*(index+1))] = alff_falff_cb
        alff_falff.append(alff_falff_cb)


        # functional connectivitry
        conn = compute_fconn(resting_ts, cerebellum_LR, targ_roi=None)
        #
        # fc[:, (91282*index):(91282*(index+1))] = conn
        fc.append(conn)

    # ALFF
    # alff_falff_mean = (alff_falff[:, 0:2] + alff_falff[:, 2:4] + alff_falff[:, 4:6] + alff_falff[:, 6:8]) / 4
    alff_falff_mean = (alff_falff[0] +  alff_falff[1] +  alff_falff[2] +  alff_falff[3]) / 4
    # save
    # Ciftiwrite(file_path=os.path.join(save_dir, 'rfMRI_REST' + run + '_cerebellum_alff_falff.dtseries.nii'), data=alff_falff_cb, cifti_ts=resting_ts, src_roi=cerebellum_LR)
    alff_falff_vol = np.zeros((91, 109, 91, alff_falff_mean.shape[1]))
    for col in np.arange(alff_falff_mean.shape[1]):
        alff_falff_vol[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2], col] = alff_falff_mean[:, col]
    save2nifti(os.path.join(save_dir, 'rfMRI_REST_cerebellum_alff_falff.nii.gz'), alff_falff_vol, affine=affine_matrix, header=None)


    # FC
    # fc_mean = (fc[:, 0:91282]+fc[:, 91282:2*91282]+fc[:, 2*91282:3*91282]+fc[:, 3*91282:4*91282]) / 4
    fc_mean = (fc[0] +  fc[1] +  fc[2] +  fc[3]) / 4
    # compute GBC
    gbc_L = compute_gbc(fc_mean, target_L)
    gbc_R = compute_gbc(fc_mean, target_R)
    # save
    # Ciftiwrite(file_path=os.path.join(save_dir, 'rfMRI_REST' + run + '_cerebellum_gbc_L.dtseries.nii'), data=gbc_L, cifti_ts=resting_ts, src_roi=cerebellum_LR)
    # Ciftiwrite(file_path=os.path.join(save_dir, 'rfMRI_REST' + run + '_cerebellum_gbc_R.dtseries.nii'), data=gbc_R, cifti_ts=resting_ts, src_roi=cerebellum_LR)
    gbc_L_vol = np.zeros((91, 109, 91, gbc_L.shape[1]))
    for gbc_L_col in np.arange(gbc_L.shape[1]):
        gbc_L_vol[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2], gbc_L_col] = gbc_L[:, gbc_L_col]
    save2nifti(os.path.join(save_dir, 'rfMRI_REST_cerebellum_gbc_L.nii.gz'), gbc_L_vol, affine=affine_matrix, header=None)

    gbc_R_vol = np.zeros((91, 109, 91, gbc_R.shape[1]))
    for gbc_R_col in np.arange(gbc_R.shape[1]):
        gbc_R_col[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2], gbc_R_col] = gbc_R[:, gbc_R_col]
    save2nifti(os.path.join(save_dir, 'rfMRI_REST_cerebellum_gbc_R.nii.gz'), gbc_R_vol, affine=affine_matrix, header=None)
