"""
    caculate ALFF of HCPD resting state data
    three step:
        1. transform time series to a frequency domain with a fast Fourier transform (FFT).
        2. calculate square root at each frequency of the power spectrum and obtaine averaged square root across 0.01–0.08 Hz at each voxel. (individual ALFF value)
        3. For standardization purposes, the ALFF of each voxel was divided by the global mean ALFF value. (about 1)
"""

import os
from scipy.fftpack import fft,ifft
import numpy as np
import pandas as pd
import nibabel as nib

# read resting state timeseries data
# subject_list = os.listdir('/nfs/e1/HCPD/fmriresults01').remove('manifests')
subject_list = ['HCD0001305_V1_MR', 'HCD0008117_V1_MR']
for subject_id in subject_list:
    run_list = ['rfMRI_REST', 'rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST1_PA']
    for run in run_list:
        img = nib.load(os.path.join('/nfs/e1/HCPD/fmriresults01', subject_id, 'MNINonLinear', 'Results', run, run + '_Atlas_MSMAll_hp0_clean.dtseries.nii'))

