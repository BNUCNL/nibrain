#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 00:36:19 2021

@author: masai
"""

import os, subprocess
import pandas as pd

# add paths
work_dir = '/nfs/e1/HCPD_CB/mri'

subject_list = pd.read_csv(os.path.join(work_dir, 'subject_info.csv'), usecols=['subID'])['subID'].tolist()
multi_model = ['vbm', 'myelination', 'alff', 'gbc']
for model in multi_model:
    print(model)
    all_vol = list()
    for subject in subject_list:
        print(subject)
        if model == 'vbm' or model == 'myelination':
            single_vol = os.path.join(work_dir, subject+'_V1_MR', 'sMRI', 'cerebellum_' + model + '_suit.nii')
            all_vol.append(single_vol)
        if model == 'alff' or model == 'gbc':
            single_vol = os.path.join(work_dir, subject+'_V1_MR', 'rfMRI', 'cerebellum_' + model + '_suit.nii')
            all_vol.append(single_vol)
    input_vol = ' '.join(all_vol)
    fslmerge_cmd = ' '.join(['fslmerge', '-t', os.path.join(work_dir, 'cerebellum_' + model + '_652sub_suit.nii.gz'), input_vol])
    subprocess.check_call(fslmerge_cmd, shell=True)

