"""
    ROI
"""

import os
import numpy as np
import nibabel as nib

# np.set_printoptions(threshold=np.inf)

# file path
hcp_roi_file = '/nfs/e2/workingshop/masai/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'
func_file = '/nfs/e1/RandomDecision/data/bold/derivatives/ciftify/sub-h27/MNINonLinear/Results/task-rdm_run-1/task-rdm_run-1_Atlas_s0.dtseries.nii'

# ROI label: R/L
# MT: 23/203
# MST: 2/182
# V4t: 156/336
# FST: 157/337
label_mt_R = 23
label_mst_R = 2
label_v4t_R = 156
label_fst_R = 157
label_mt_L = 203
label_mst_L = 182
label_v4t_L = 336
label_fst_L = 337

# load HCP ROI file
roi_img = nib.load(hcp_roi_file)
roi_data = roi_img.get_fdata() # shape: (1, 59412)
roi_header = roi_img.header

# load func data
func_img = nib.load(func_file)
func_data = func_img.get_fdata()
func_header = func_img.header

# extract and integrate ROIs
roi_array = np.squeeze(roi_data)
for index, label in enumerate(roi_array):
    if label == label_mt_R or label == label_mst_R or label == label_v4t_R or label == label_fst_R or label == label_mt_L or label == label_mst_L or label == label_v4t_L or label == label_fst_L:
        continue
    else:
        roi_array[index] = 0
roi_data = np.expand_dims(roi_array, axis=1).reshape(1,-1)

# save new roi
new_roi_img = nib.Cifti2Image(roi_data, roi_header)
nib.cifti2.save(new_roi_img, '/nfs/e2/workingshop/masai/rdmdec_workdir/new_roi.dlabel.nii')

# mask func data
for index, label in enumerate(roi_array):
    if not label == 0:
        continue
    else:
        for surf in func_data:
            surf[index] = 0

# save new func
new_func_img = nib.Cifti2Image(func_data, func_header)
nib.cifti2.save(new_func_img, '/nfs/e2/workingshop/masai/rdmdec_workdir/task-rdm_run-1_Atlas_s0&roi.dtseries.nii')














