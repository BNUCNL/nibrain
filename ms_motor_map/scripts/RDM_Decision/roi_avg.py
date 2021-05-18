"""
    roi_avg
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# file path
rootPath = '/nfs/e2/workingshop/masai/rdmdec_workdir'
savePath = '/nfs/e2/workingshop/masai/rdmdec_workdir/roi_avg'

# read trail time series
behave_df = pd.read_csv(os.path.join(rootPath, 'sub-h27.csv'))
onset_df = behave_df[['osti', 'oiti']].loc[0:49]
for index, row in onset_df.iterrows():


# for i in onset:
#     print(i)


# iterate every run
allfile = os.listdir(rootPath)
for file in allfile:
    if 's0&roi.dtseries.nii' in file:
        print(file)
        # load func data
        func_img = nib.load(rootPath + '/' + file)
        func_data = func_img.get_fdata()
        func_header = func_img.header

        avg_all = []
        shape = func_data.shape
        cnt = np.count_nonzero(func_data[0])
        for i in range(0, shape[0]):
            avg = np.sum(func_data[i]) / cnt
            func_data[i] = np.where(func_data[i], np.ones(shape[1])*avg, 0)
            avg_all.append(avg)

        # save new func
        new_func_img = nib.Cifti2Image(func_data, func_header)
        nib.cifti2.save(new_func_img, savePath + '/' + 'AVG' + file)

        run_id = file[13]
        plt.title('ROI Average of run' + str(run_id))
        plt.plot(range(0,shape[0]), avg_all, color='green')

        plt.xlabel('time')
        plt.ylabel('roi_avg')
        plt.show()

        print(file + ' finished!')