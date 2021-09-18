import os, shutil
import pandas as pd

work_dir = '/nfs/e1/HCPD_CB/mri/'
subject_list = pd.read_csv(os.path.join(work_dir, 'subject_list.csv'), header=None)[0].tolist()

for subject in subject_list:

    shutil.rmtree(os.path.join(work_dir, subject, 'func'))
    if os.path.exists(os.path.join(work_dir, subject, 'MNINonLinear')):
        shutil.rmtree(os.path.join(work_dir, subject, 'MNINonLinear'))
    shutil.rmtree(os.path.join(work_dir, subject, 'rfMRI'))

    sMRI = os.path.join(work_dir, subject, 'sMRI')
    os.mkdir(sMRI)
    rfMRI = os.path.join(work_dir, subject, 'rfMRI')
    os.mkdir(rfMRI)
    tfMRI = os.path.join(work_dir, subject, 'tfMRI')
    os.mkdir(tfMRI)
    dMRI = os.path.join(work_dir, subject, 'dMRI')
    os.mkdir(dMRI)

    old_anat_path = os.path.join(work_dir, subject, 'anat')
    shutil.move(os.path.join(old_anat_path, 'c_T1w.nii'), os.path.join(sMRI, 'cerebellum_cropped_native.nii'))
    shutil.move(os.path.join(old_anat_path, 'c_T1w_pcereb.nii'), os.path.join(sMRI, 'cerebellum_mask_native.nii'))
    shutil.move(os.path.join(old_anat_path, 'T1w_seg1.nii'), os.path.join(sMRI, 'cerebellum_graymatter_prob_native.nii'))
    shutil.move(os.path.join(old_anat_path, 'T1w_seg2.nii'), os.path.join(sMRI, 'cerebellum_whitematter_prob_native.nii'))
    shutil.move(os.path.join(old_anat_path, 'Affine_T1w_seg1.mat'), os.path.join(sMRI, 'cerebellum_normalize_affine_linear.mat'))
    shutil.move(os.path.join(old_anat_path, 'u_a_T1w_seg1.nii'), os.path.join(sMRI, 'cerebellum_normalize_flowfield_nonlinear.nii'))
    shutil.move(os.path.join(old_anat_path, 'wdmyelin.nii'), os.path.join(sMRI, 'cerebellum_myelination_suit.nii'))
    shutil.move(os.path.join(old_anat_path, 'wdT1w_seg1.nii'), os.path.join(sMRI, 'cerebellum_vbm_suit.nii'))

    shutil.rmtree(os.path.join(work_dir, subject, 'anat'))




