"""
    remove brainstem from SUIT template using
"""

import os
import numpy as np
import nibabel as nib

# read mask data
mask_data = nib.load('/usr/local/neurosoft/matlab_tools/spm12/toolbox/suit/atlasesSUIT/Lobules-SUIT.nii').get_fdata()

# Loop: process myelin & vbm
subject_list = os.listdir('/nfs/e1/HCPD_CB/mri')
for subj in subject_list:
    # read data, header and affine
    myelin_img = nib.load(os.path.join('/nfs/e1/HCPD_CB/mri/', subj, 'anat', 'wdmyelin_map.nii'))
    myelin_header = myelin_img.header
    myelin_data = np.nan_to_num(myelin_img.get_fdata())
    myelin_affine = myelin_img.affine
    vbm_img = nib.load(os.path.join('/nfs/e1/HCPD_CB/mri/', subj, 'anat', 'wdT1w_seg1.nii'))
    vbm_header = vbm_img.header
    vbm_data = np.nan_to_num(vbm_img.get_fdata())
    vbm_affine = vbm_img.affine
    # mask
    myelin_data_masked = np.where(myelin_data * mask_data == 0, 0, myelin_data)
    vbm_data_masked = np.where(vbm_data * mask_data == 0, 0, vbm_data)
    # save
    myelin_image_masked = nib.Nifti1Image(myelin_data_masked, affine=myelin_affine, header=myelin_header)
    nib.save(myelin_image_masked, os.path.join('/nfs/e1/HCPD_CB/mri/', subj, 'anat', 'myelin_map_masked.nii'))
    vbm_image_masked = nib.Nifti1Image(vbm_data_masked, affine=vbm_affine, header=vbm_header)
    nib.save(vbm_image_masked, os.path.join('/nfs/e1/HCPD_CB/mri/', subj, 'anat', 'vbm_masked.nii'))
