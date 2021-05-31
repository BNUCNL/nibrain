"""
    copy design matrix from old motormapping to denoise validation folder
    /nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify/sub-M02/MNINonLinear/Results/ses-01_task-motor_run-3/ses-01_task-motor_run-3_hp200_s4_level1.feat/
"""

import os, shutil, re

old_bids_dir = '/nfs/e4/function_guided_resection/MotorMapping'
new_bids_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/nifti'

old_cifti_dir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify'
new_matrix_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/design_matrix'

subjid_list = [sub.replace("sub-", "") for sub in os.listdir(new_bids_dir) if "sub-" in sub]
subjid_list.remove('01')

for subjid in subjid_list:
    # /nfs/e4/function_guided_resection/MotorMapping/sub-M01/ses-01/func/
    old_func_dir = os.path.join(old_bids_dir, 'sub-M' + subjid, 'ses-01', 'func')
    # /nfs/e4/function_guided_resection/MotorMap/data/bold/nifti/sub-01/ses-1/func/
    new_func_dir = os.path.join(new_bids_dir, 'sub-' + subjid, 'ses-1', 'func')
    # reindex old run id
    old_runid_list = []
    for filename in os.listdir(old_func_dir):
        if '_events.tsv' in filename:
            old_runid_list.append(str(re.findall('run-(.+?)_events', filename)[0]))
    old_runid_list.sort(key=int)
    # print(old_runid_list)
    runid_dict = {v: str(k+1) for k, v in dict(enumerate(old_runid_list)).items()}
    # print(runid_dict)
    for old_runid, new_runid in runid_dict.items():
        print('*****************************************')
        print('subject: ' + subjid)
        print('old:' + old_runid + '--new:' + new_runid)
        design_mat = os.path.join(old_cifti_dir, 'sub-M' + subjid, 'MNINonLinear', 'Results', 'ses-01_task-motor_run-' + old_runid, 'ses-01_task-motor_run-' + old_runid + '_hp200_s4_level1.feat', 'design.mat')
        target_dir = os.path.join(new_matrix_dir, 'sub-' + subjid, 'run-' + new_runid)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copyfile(design_mat, os.path.join(target_dir, 'design.mat'))






