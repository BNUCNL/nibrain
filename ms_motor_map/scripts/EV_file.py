import os
import pandas as pd
import numpy as np
import nibabel as nib

# EV
subject_list = pd.read_csv()





# coverage map 1 prob_map
subject_list = pd.read_csv().to_list()
subject_list.remove('sub-04')
subject_list.remove('sub-23')
subject_list.remove('sub-27')
run_list = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6']

map = np.zeros((1, 91282))
for subject in subject_list:
    for run in run_list:
        data = nib.load(os.path.join()).get_fdata()
        data[data != 0] = 1
        map = map + data

prob_map = map / len(subject_list)
nib.save(nib.Cifti2Image(prob_map, nib.load('').header), '')

# coverage map 2 bar plot

