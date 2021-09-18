#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 03:51:34 2021

@author: masai
"""

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nibabel as nib
from ioTools import save2nifti

# calculate mean
def network_mean(map, atlas):
    mean_array = np.zeros((int(atlas.max()),))
    for network in np.arange(1, int(atlas.max())+1):
        mean_array[network-1,] = np.mean(map[atlas==network])
    return mean_array

# calculate dice
def dice_coeff(map, atlas):
    map = abs(map)
    dice_array = np.zeros((int(atlas.max()),))
    for network in np.arange(1, int(atlas.max()) + 1):
        mask = (atlas == network).astype(int)
        dice_array[network-1,] = (2 * np.sum(map * mask)) / (np.sum(map) + np.sum(mask))
    return dice_array

# load atlas & set components path
suit_path = '/usr/local/neurosoft/matlab_tools/spm12/toolbox/suit/atlasesSUIT/'
buckner_17networks = nib.load(os.path.join(suit_path, 'Buckner_17Networks.nii')).get_fdata()
buckner_7networks = nib.load(os.path.join(suit_path, 'Buckner_7Networks.nii')).get_fdata()
components_path = '/nfs/z1/userhome/MaSai/workingdir/code/cb/test_res/pca_test/652sub/'
cb_mask = nib.load('/nfs/z1/userhome/MaSai/workingdir/code/cb/test_res/cerebellum_mask.nii').get_fdata().astype(bool)


# plot mean
for component in np.arange(4):
    # load component
    component_map = nib.load(os.path.join(components_path, 'components_000'+str(component)+'.nii')).get_fdata()
    plt.subplot(2, 2, component+1)
    y = network_mean(component_map, buckner_7networks)
    x = np.arange(1, y.shape[0]+1)
    plt.bar(x, abs(y))
    plt.xticks(x)
    plt.xlabel('network id')
    plt.ylabel('gradient mean (absolute)')
    plt.title('component'+str(component+1)+'-buckner7networks')
    for i in x-1:
        plt.text(x[i], abs(y[i]), np.around(y[i], 3), fontsize=10, ha='center', va='bottom')
plt.show()

# plot dice
for component in np.arange(4):
    # load component
    component_map = nib.load(os.path.join(components_path, 'components_000'+str(component)+'.nii')).get_fdata()
    component_map[component_map > (np.sum(component_map) / np.count_nonzero(component_map))] = 0

    plt.subplot(2, 2, component+1)
    y = dice_coeff(component_map, buckner_7networks)
    x = np.arange(1, y.shape[0]+1)
    plt.bar(x, y)
    plt.xticks(x)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    plt.xlabel('network id')
    plt.ylabel('dice coefficient')
    plt.title('component'+str(component+1)+'-buckner7networks threshold(gradient<mean)')
    for i in x-1:
        plt.text(x[i], y[i], np.around(y[i], 3), fontsize=10, ha='center', va='bottom')
plt.show()