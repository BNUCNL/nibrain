#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 03:17:24 2021

@author: MaSai
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib


def tSNR(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_tSNR = np.nan_to_num(data_mean / data_std)
    return data_tSNR

def cohen_d(pre, post):
    npre = np.shape(pre)[-1]
    npost = np.shape(post)[-1]
    dof = npost + npre - 2
    d = ((post.mean(-1) - pre.mean(-1)) /
         np.sqrt(((npost - 1) * np.var(post, axis=-1, ddof=1) +
                  (npre - 1) * np.var(pre, axis=-1, ddof=1)) / dof))
    d = np.nan_to_num(d)
    return d


subject_list = pd.read_csv('/nfs/z1/userhome/MaSai/workingdir/code/motor/subject_list.csv', header=None)[0].to_list()
# subject_list = ['sub-01']

# raw tSNR
raw_data_dir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify'
raw_res_dir = '/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/denoise_validation/comparison/raw'
header = nib.load(
    '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify/sub-M01/MNINonLinear/Results/ses-01_task-motor/ses-01_task-motor_hp200_s4_level2.feat/sub-M01_ses-01_task-motor_level2_zstat_Finger-Avg_hp200_s4.dscalar.nii').header
# across subs & runs
for subject in subject_list:
    subject_id = 'sub-M' + subject[-2:]
    run_list = os.listdir(os.path.join(raw_data_dir, subject_id, 'MNINonLinear', 'Results'))
    run_list.remove('ses-01_task-motor')
    tSNR_sub = np.zeros((91282,))
    for run in run_list:
        data = nib.load(os.path.join(raw_data_dir, subject_id, 'MNINonLinear', 'Results', run,
                                    run + '_Atlas_hp200_s4.dtseries.nii')).get_fdata()
        tSNR_run = tSNR(data)
        tSNR_sub = tSNR_sub + tSNR_run
    tSNR_sub = tSNR_sub / len(run_list)
    tSNR_sub = tSNR_sub.reshape(1, -1)
    nib.save(nib.Cifti2Image(tSNR_sub, header), os.path.join(raw_res_dir, subject + '_tSNR.dscalar.nii'))
    print(subject + ' done!')
# each movement conditions
condition_dict = {
    1:'toe',
    2:'ankle',
    3:'leftleg',
    4:'rightleg',
    5:'forearm',
    6:'upperarm',
    7:'wrist',
    8:'finger',
    9:'eye',
    10:'jaw',
    11:'lip',
    12:'tongue'
}
for condition in range(11, 13):
    print(condition)
    tSNR_condition = np.zeros((91282,))
    for subject in subject_list:
        print(subject)
        subject_id = 'sub-M' + subject[-2:]
        run_list = os.listdir(os.path.join(raw_data_dir, subject_id, 'MNINonLinear', 'Results'))
        run_list.remove('ses-01_task-motor')
        for run in run_list:
            print(run)
            data_all = nib.load(os.path.join(raw_data_dir, subject_id, 'MNINonLinear', 'Results', run, run + '_Atlas_hp200_s4.dtseries.nii')).get_fdata()
            events_file = os.path.join('/nfs/e4/function_guided_resection/MotorMapping', subject_id, 'ses-01', 'func', subject_id + '_' + run + '_events.tsv')
            events_df = pd.read_csv(events_file, sep='\t')
            condition_onset_df = events_df[events_df['trial_type'] == condition]['onset']
            for onset in condition_onset_df:
                onset_index = int(onset / 2) # TR = 2s
                data = data_all[onset_index:onset_index+8, :]
                tSNR_single_condition = tSNR(data)
                tSNR_condition = tSNR_condition + tSNR_single_condition
    tSNR_condition = tSNR_condition / 744
    tSNR_condition = tSNR_condition.reshape(1, -1)
    nib.save(nib.Cifti2Image(tSNR_condition, header), os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/denoise_validation/conditions', condition_dict[condition] + '_tSNR.dscalar.nii'))
    print(condition_dict[condition]+' DONE!')

# denised tSNR
denoised_data_dir = '/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify'
denoised_res_dir = '/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/denoise_validation/comparison/denoised'
for subject in subject_list:
    run_list = os.listdir(os.path.join(raw_data_dir, subject, 'MNINonLinear', 'Results'))
    tSNR_sub = np.zeros((91282,))
    for run in run_list:
        img = nib.load(os.path.join(raw_data_dir, subject, 'MNINonLinear', 'Results', run,
                                    run + '_Atlas_hp200_s4.dtseries.nii'))
        data = img.get_fdata()
        tSNR_run = tSNR(data)
        tSNR_sub = tSNR_sub + tSNR_run
    tSNR_sub = tSNR_sub / len(run_list)
    tSNR_sub = tSNR_sub.reshape(1, -1)
    nib.save(nib.Cifti2Image(tSNR_sub, header), os.path.join(raw_res_dir, subject + '_tSNR.dscalar.nii'))





# visulization
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

condition = 'rightleg'
data = nib.load('ms_motor_map/data/conditions/'+condition+'_tSNR.dscalar.nii').get_fdata()

plt.hist(data[0,:], bins=50, rwidth=0.5)
plt.title('tSNR of '+condition)
plt.xlabel('tSNR')
plt.ylabel('Number of vertices')
plt.show()
