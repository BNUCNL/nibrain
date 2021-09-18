#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:25:05 2021

@author: masai
"""

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nibabel as nib
from ioTools import save2nifti

# load component
components_path = '/nfs/z1/userhome/MaSai/workingdir/code/cb/test_res/pca_test/652sub/'
component_img = nib.load(os.path.join(components_path, 'components_0000.nii'))
component_data = component_img.get_fdata()
cb_mask = nib.load('/nfs/z1/userhome/MaSai/workingdir/code/cb/test_res/cerebellum_mask.nii').get_fdata().astype(bool)
component_data_flat = component_data[cb_mask].reshape(-1,1)

cluster_num = 5
for i in range(1, 11):
    # kmeans model
    component_cluster = KMeans(n_clusters=cluster_num, random_state=i).fit_predict(component_data_flat)
    # save
    save_vol = np.zeros((141, 95, 87))
    save_vol[cb_mask] = component_cluster+1
    save2nifti('/nfs/z1/userhome/MaSai/workingdir/code/cb/test_res/cluster/kmeans/component1_'+str(cluster_num)+'clusters_'+str(i)+'.nii.gz', save_vol, affine=component_img.affine)