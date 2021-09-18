#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 02:22:46 2021

@author: masai
"""

import os, subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from ioTools import save2nifti

# add paths
# add paths
work_dir = '/nfs/e1/HCPD_CB/mri/'
suit_atlas_dir = '/usr/local/neurosoft/matlab_tools/spm12/toolbox/suit/atlasesSUIT'
# read subject id list
subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None).iloc[:,0].tolist()
# Loop: process myelin & vbm
for subject in subject_list:

    print(subject)

    sMRI_dir = os.path.join(work_dir, subject, 'sMRI')
    rfMRI_dir = os.path.join(work_dir, subject, 'rfMRI')
    # mask vbm
    suit_atlas_input = os.path.join(suit_atlas_dir, 'Lobules-SUIT.nii')
    vbm_input = os.path.join(sMRI_dir, 'cerebellum_vbm_suit.nii')
    vbm_output = os.path.join(sMRI_dir, 'cerebellum_vbm_masked_suit.nii.gz')
    fslmaths_cmd_vbm = ' '.join(['fslmaths', vbm_input, '-mas', suit_atlas_input, vbm_output])
    try:
        subprocess.check_call(fslmaths_cmd_vbm, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('VBM: Error happened in subject {}'.format(subject))

subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None).iloc[:,0].tolist()
all_vol = list()
for subject in subject_list:
    single_vol = os.path.join(work_dir, subject, 'sMRI', 'cerebellum_vbm_masked_suit.nii.gz')
    all_vol.append(single_vol)
input_vol = ' '.join(all_vol)
fslmerge_cmd = ' '.join(['fslmerge', '-t', os.path.join(work_dir, 'cerebellum_mask_652sub_suit.nii.gz'), input_vol])
subprocess.check_call(fslmerge_cmd, shell=True)

#
vbm_ts = nib.load(os.path.join(work_dir, 'cerebellum_mask_652sub_suit.nii.gz')).get_fdata()
cb_mask = np.all(vbm_ts, axis=3).astype(int)
affine_matrix = nib.load('/nfs/e1/HCPD_CB/mri/HCD0001305_V1_MR/sMRI/cerebellum_vbm_suit.nii').affine
save2nifti(os.path.join(work_dir, 'cerebellum_mask_652sub_suit.nii.gz'), cb_mask, affine=affine_matrix, header=None)

#
cb_mask = os.path.join(work_dir, 'cerebellum_mask_652sub_suit.nii.gz')
for subject in subject_list:
    print(subject)
    sMRI_dir = os.path.join(work_dir, subject, 'sMRI')
    rfMRI_dir = os.path.join(work_dir, subject, 'rfMRI')
    # mask vbm
    vbm_input = os.path.join(sMRI_dir, 'cerebellum_vbm_suit.nii')
    vbm_output = os.path.join(sMRI_dir, 'cerebellum_vbm_masked_suit.nii.gz')
    fslmaths_cmd_vbm = ' '.join(['fslmaths', vbm_input, '-mas', cb_mask, vbm_output])
    try:
        subprocess.check_call(fslmaths_cmd_vbm, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('VBM: Error happened in subject {}'.format(subject))
    # mask myelin
    myelin_input = os.path.join(sMRI_dir, 'cerebellum_myelination_suit.nii')
    myelin_output = os.path.join(sMRI_dir, 'cerebellum_myelination_masked_suit.nii.gz')
    fslmaths_cmd_myelin = ' '.join(['fslmaths', myelin_input, '-mas', cb_mask, myelin_output])
    try:
        subprocess.check_call(fslmaths_cmd_myelin, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('MYELIN: Error happened in subject {}'.format(subject))
    # mask alff
    alff_input = os.path.join(rfMRI_dir, 'cerebellum_alff_suit.nii')
    alff_output = os.path.join(rfMRI_dir, 'cerebellum_alff_masked_suit.nii.gz')
    fslmaths_cmd_alff = ' '.join(['fslmaths', alff_input, '-mas', cb_mask, alff_output])
    try:
        subprocess.check_call(fslmaths_cmd_alff, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('ALFF: Error happened in subject {}'.format(subject))
    # mask gbc
    gbc_input = os.path.join(rfMRI_dir, 'cerebellum_gbc_suit.nii')
    gbc_output = os.path.join(rfMRI_dir, 'cerebellum_gbc_masked_suit.nii.gz')
    fslmaths_cmd_gbc = ' '.join(['fslmaths', gbc_input, '-mas', cb_mask, gbc_output])
    try:
        subprocess.check_call(fslmaths_cmd_gbc, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('GBC: Error happened in subject {}'.format(subject))



