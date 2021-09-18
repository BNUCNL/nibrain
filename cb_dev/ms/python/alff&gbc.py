#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:16:49 2021

@author: masai
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from HCPatlas import CAP
from ioTools import CiftiReader, save2nifti

hcpd_dir = '/nfs/z1/HCP/HCPD/fmriresults01/'
work_dir = '/nfs/e1/HCPD_CB/mri/'

# cerebellum volume index & affine
resting_img = CiftiReader('/nfs/e1/HCPD/fmriresults01/HCD0008117_V1_MR/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii')
affine_matrix = resting_img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
cb_L_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT', zeroize=False)[2]
cb_R_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_RIGHT', zeroize=False)[2]
cb_L_index.extend(cb_R_index)
cb_index_matrix = np.asarray(cb_L_index)
vol_size = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT', zeroize=False)[1]

cap = CAP('/nfs/e2/workingshop/masai/code/cb/atlas/ColeAnticevicNetPartition')
cerebellum_mask = cap.get_structure(['Cerebellum'])
parcel_in_cc = cap.get_structure(['Ctx'], return_idx=True)
paecel_in_networks = cap.get_network(return_idx=True)

subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None)[0].tolist()
for subject in subject_list:

    print(subject)

    rfmri_dir = os.path.join(work_dir, subject, 'rfMRI')
    res_dir = os.path.join(hcpd_dir, subject, 'MNINonLinear', 'Results')

    # ALFF & fALFF
    alff_cb = nib.load(os.path.join(res_dir, 'alff.dscalar.nii')).get_fdata()[:, cerebellum_mask[0, :]]
    alff_vol = np.zeros(vol_size)
    alff_vol[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2]] = alff_cb[0, :]
    save2nifti(os.path.join(rfmri_dir, 'cerebellum_alff_mni.nii.gz'), alff_vol, affine=affine_matrix, header=None)
    falff_cb = nib.load(os.path.join(res_dir, 'falff.dscalar.nii')).get_fdata()[:, cerebellum_mask[0, :]]
    falff_vol = np.zeros(vol_size)
    falff_vol[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2]] = falff_cb[0, :]
    save2nifti(os.path.join(rfmri_dir, 'cerebellum_falff_mni.nii.gz'), falff_vol, affine=affine_matrix, header=None)

    # GBC
    fc_cb = nib.load(os.path.join(res_dir, 'rsfc_ColeParcel2Vertex.dscalar.nii')).get_fdata()[:, cerebellum_mask[0, :]]
    fc_cb_cc = np.mean(fc_cb[parcel_in_cc[0],:], axis=0)
    fc_cb_cc_vol = np.zeros(vol_size)
    fc_cb_cc_vol[cb_index_matrix[:, 0], cb_index_matrix[:, 1], cb_index_matrix[:, 2]] = fc_cb_cc
    save2nifti(os.path.join(rfmri_dir, 'cerebellum_cortex_gbc_mni.nii.gz'), falff_vol, affine=affine_matrix, header=None)























