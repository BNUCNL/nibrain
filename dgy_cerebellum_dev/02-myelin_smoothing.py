import os
import tqdm
import nibabel
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

import lib
plt.style.use('ggplot')

# Define ROI
ROI = lib.roi.VolumeWMParc()
indices = ROI.get_idx(ROI.get_roi_id('CEREBELLUM-CORTEX'))
exid = ROI.get_idx(ROI.get_roi_id('CEREBELLUM-CORTEX'), exclude_specified_id=True)

def generate_smoothed_map():
    sub_list = lib.basic.SUB_AVAILABLE
    progress_bar = tqdm.tqdm(total=len(sub_list))
    for sub_id in sub_list:
        progress_bar.update(1)
        if os.path.isfile(f'temp/{sub_id}_myelin_map.nii.gz'):
            continue
        
        myelin_map = lib.myelin.get_vol_map(sub_id)
        nib_img = lib.niio.get_vol_obj(myelin_map)
        nibabel.save(nib_img, f'temp/{sub_id}_myelin_map.nii.gz')

        cmd = f'fslmaths temp/{sub_id}_myelin_map.nii.gz -s {4 / 2.355:.2f} temp/Smoothed_{sub_id}_myelin_map.nii.gz'
        os.system(cmd)

def generate_mean_map():
    sub_list = lib.basic.SUB_AVAILABLE
    mean_smoothed = np.row_stack(
        [nibabel.load(f'temp/Smoothed_{sub_id}_myelin_map.nii.gz').get_fdata()[indices] for sub_id in sub_list]
    ).mean(axis=0)
    nib_img = lib.niio.get_vol_obj(ROI.invert(mean_smoothed, ROI.get_roi_id('CEREBELLUM-CORTEX')))
    nibabel.save(nib_img, f'temp/mean_smoothed_myelin_map.nii.gz')

def plot_mean_map_hist():
    nib_img = nibabel.load(f'temp/mean_smoothed_myelin_map.nii.gz')
    data = nib_img.get_fdata()[indices]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(data[data <= 3], bins=15, color='#2980b9')
    axes[0].set_title('data <= 3')
    axes[1].hist(data[data > 3], bins=15, color='#9b59b6')
    axes[1].set_title('data > 3')
    plt.savefig('temp/hist_separated_by_3.svg')
    plt.show()

def export(fname='data/myelin_grad/cb_mask.nii.gz'):
    nib_img = nibabel.load(f'temp/mean_smoothed_myelin_map.nii.gz')
    data = nib_img.get_fdata()
    threshold = 3
    data[data <= threshold] = 1
    data[data >= threshold] = 0
    nib_img = lib.niio.get_vol_obj(data)
    nibabel.save(nib_img, fname)

# Pipeline
if __name__ == '__main__':
    generate_smoothed_map()
    generate_mean_map()
    plot_mean_map_hist()
    export()