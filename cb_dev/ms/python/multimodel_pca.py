#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 02:22:46 2021

@author: masai
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from ioTools import save2nifti
from scipy.stats.stats import sem

# add paths
work_dir = '/nfs/e1/HCPD_CB/mri/'

cb_mask = nib.load('/nfs/e2/workingshop/masai/code/cb/test_res/cerebellum_mask.nii').get_fdata().astype(bool)
# load data
vbm = nib.load(os.path.join(work_dir, 'cerebellum_vbm_652sub_suit.nii.gz')).get_fdata()[cb_mask]
myelin = nib.load(os.path.join(work_dir, 'cerebellum_myelination_652sub_suit.nii.gz')).get_fdata()[cb_mask]
alff = nib.load(os.path.join(work_dir, 'cerebellum_alff_652sub_suit.nii.gz')).get_fdata()[cb_mask]
gbc = nib.load(os.path.join(work_dir, 'cerebellum_gbc_652sub_suit.nii.gz')).get_fdata()[cb_mask]

multimodel_matrix = np.concatenate((vbm, myelin, alff, gbc), axis=1)
multimodel_matrix_mean = np.sum(multimodel_matrix, axis=0) / np.count_nonzero(multimodel_matrix, axis=0)
multimodel_matrix[:, np.arange(multimodel_matrix.shape[1])] = np.where(multimodel_matrix[:, np.arange(multimodel_matrix.shape[1])] == 0, multimodel_matrix_mean[np.arange(multimodel_matrix.shape[1]),], multimodel_matrix[:, np.arange(multimodel_matrix.shape[1])])
multimodel_matrix_zstat = preprocessing.scale(multimodel_matrix, axis=0)

pca = PCA(n_components=10)
pca.fit(multimodel_matrix_zstat)
pca.explained_variance_ratio_

plt.figure()
x = np.arange(10)
y = pca.explained_variance_ratio_
xticklabels = np.arange(1, 11)
plt.plot(x, y)
plt.xticks(x, xticklabels)
plt.title('explained variance ratio of 10 components')
plt.xlabel('component')
plt.ylabel('explained variance ratio')
plt.show()

component_map = pca.transform(multimodel_matrix_zstat)
save_vol = np.zeros((141, 95, 87, 10))
save_vol[cb_mask] = component_map
save2nifti('/nfs/e2/workingshop/masai/code/cb/test_res/pca_test/652sub/components_map.nii.gz', save_vol, affine=nib.load('/nfs/e2/workingshop/masai/code/cb/test_res/cerebellum_mask.nii').affine)

component_age_series = pca.components_
subject_info = pd.read_csv(os.path.join(work_dir, 'subject_info.csv'), usecols=['subID', 'age in months', 'age in years'])

# component_id = 1
# model_list = ['VBM', 'Myelin', 'ALFF', 'GBC']
# for idx, model in enumerate(model_list):
#     plt.subplot(2, 2, idx+1)
#     x = np.array(subject_info['age in years'])
#     y = component_age_series[(component_id-1), (652*idx):(652*idx+652)]
#     plt.scatter(x, y)
#     z = np.polyfit(x, y, 2)
#     p = np.poly1d(z)
#     plt.plot(x, p(x), linestyle='--')
#     xticklabels = np.array(subject_info['age in years'])
#     plt.xticks(x, xticklabels)
#     plt.title('component-'+str(component_id)+'_'+model)
#     plt.xlabel('age in years')
#     plt.ylabel('weight')
# plt.show()

age_array = np.array(subject_info['age in years'])

component_id = 1
model_list = ['VBM', 'Myelin', 'ALFF', 'GBC']
for idx, model in enumerate(model_list):
    x = subject_info['age in years'].unique()
    y = np.zeros((17,))
    y_error = np.zeros((17,))
    for i, age in enumerate(range(6, 23)):
        age_matrix = component_age_series[(component_id - 1), (652 * idx):(652 * idx + 652)][age_array==age]
        y[i,] = np.mean(age_matrix)
        y_error[i,] = sem(age_matrix)
    plt.errorbar(x, y, yerr=y_error, label=model)
    xticklabels = subject_info['age in years'].unique()
    plt.xticks(x, xticklabels)
    plt.title('component-'+str(component_id))
    plt.xlabel('age in years')
    plt.ylabel('weight')
plt.legend(loc='upper right')