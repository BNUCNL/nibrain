#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:52:48 2021

@author: masai
"""

# %%
import os 
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from ioTools import CiftiReader, save2nifti
from scipy import signal

# %%
# test path
test_file = '/nfs/e1/HCPD/fmriresults01/HCD0008117_V1_MR/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii'

# %%
# read subjects list
subject_list = pd.read_csv('/nfs/e1/HCPD_CB/mri/subject_list.csv', header=None).values.tolist()

for subject in subject_list:
    
    # load functional data & ROIs
    cifti_img = CiftiReader(os.path.join())
    cerebellum_L_data = cifti_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT', zeroize=False) # (478, 8709)
    cerebellum_R_data = cifti_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_RIGHT', zeroize=False) # (478, 9144)
    
    # organize the data into matrix style
    cb_voxel_matrix = np.concatenate((cerebellum_L_data[0], cerebellum_R_data[0]), axis=1) # (478, 17853)
    
    # caculate alff & falff
    
    # takes fast Fourier transform of timeseries and calculates frequency scale
    fft_array = fft(signal.detrend(cb_voxel_matrix, axis=0, type='linear'), axis=0)
    freq_scale = np.fft.fftfreq(cb_voxel_matrix.shape[0], 0.8)
    
    # Calculates power of fft (0-0.625Hz)
    total_power = np.sqrt(np.absolute(fft_array[(0.0 <= freq_scale) & (freq_scale <= 0.625), :]))
    
    # calculates alff & falff
    alff = np.sum(total_power[(0.008 <= freq_scale)[0:239] & (freq_scale <= 0.1)[0:239], :], axis=0)
    falff = alff / np.sum(total_power, axis=0)
    
    # save results as 3D nifti format
    
    # create matrix to save data
    alff_matrix = np.zeros(cerebellum_L_data[1])
    falff_matrix = np.zeros(cerebellum_L_data[1])
    
    # fill values by cb_L/R_data index
    cerebellum_L_data[2].extend(cerebellum_R_data[2])
    cb_index_matrix = np.asarray(cerebellum_L_data[2])
    alff_matrix[cb_index_matrix[:,0],cb_index_matrix[:,1],cb_index_matrix[:,2]] = alff
    falff_matrix[cb_index_matrix[:,0],cb_index_matrix[:,1],cb_index_matrix[:,2]] = falff
    
    # save as nifti
    affine_matrix = cifti_img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
    save2nifti('/nfs/e2/workingshop/masai/code/cb/python/alff.nii.gz', alff_matrix, affine=affine_matrix, header=None)
    save2nifti('/nfs/e2/workingshop/masai/code/cb/python/falff.nii.gz', falff_matrix, affine=affine_matrix, header=None)

