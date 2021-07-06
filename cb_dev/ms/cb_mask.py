"""
    remove brainstem from SUIT template using fslmaths command
"""

import os, subprocess
import pandas as pd

# add paths
work_dir = '/nfs/e1/HCPD_CB/mri/'
suit_atlas_dir = '/usr/local/neurosoft/matlab_tools/spm12/toolbox/suit/atlasesSUIT'
# read subject id list
subject_id_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None).iloc[:,0].tolist()
# Loop: process myelin & vbm
for subject_id in subject_id_list:
    # input img & output img
    vbm_img_input = os.path.join(work_dir, subject_id, 'anat', 'wdT1w_seg1.nii')
    myelin_img_input = os.path.join(work_dir, subject_id, 'anat', 'wdmyelin.nii')
    suit_atlas_input = os.path.join(suit_atlas_dir, 'Lobules-SUIT.nii')
    vbm_img_output = os.path.join(work_dir, subject_id, 'anat', 'vbm_masked.nii')
    myelin_img_output = os.path.join(work_dir, subject_id, 'anat', 'myelin_masked.nii')
    # mask vbm
    fslmaths_cmd_vbm = ' '.join(['fslmaths', vbm_img_input, '-mas', suit_atlas_input, vbm_img_output])
    try:
        subprocess.check_call(fslmaths_cmd_vbm, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('CB_MASK: Error happened in subject {}'.format(subject_id))
    # mask myelin
    fslmaths_cmd_myelin = ' '.join(['fslmaths', myelin_img_input, '-mas', vbm_img_output, myelin_img_output])
    try:
        subprocess.check_call(fslmaths_cmd_myelin, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('CB_MASK: Error happened in subject {}'.format(subject_id))