"""
calculate the tSNR for pre- and post-denoising fMRI data
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from collections import Counter

sessidlist = ["sub001"]


runidlist = ["001"]


# smth_cdt = 'unsmth'
# hemi = 'rh'
# projectdir = "/nfs/e5/studyforrest"
# after_funcname = "audiovisual3T_fslpreproc2surface_{0}_denoised".format(smth_cdt)
# before_funcname = "audiovisual3T_fslpreproc2surface_{0}".format(smth_cdt)
# file_name = 'fmcpr.sm0.fsaverage.{0}.nii.gz'.format(hemi)
#
# analysis_dir = '/nfs/e5/studyforrest/data_lxy/result/tSNR/surface'
# figure_dir = '/nfs/e5/studyforrest/data_lxy/result/tSNR/figure'
vertex_number = 163842


def tSNR(data):
    """alculate the temporal signal-to-noise ratio (tSNR) for each vertex
    Parameters
    ----------          
 
        data: used to calculate tSNR, 
            shape = [n_vertice, m_timepoints].
    
    Returns
    -------
        data_tSNR: the tSNR of data, shape = [n_vertice, ].
   
    Notes
    -----
		The tSNR was defined as the ratio between the mean of a timeseries 
		and its SD for each vertex
    """
    
    data_mean = np.mean(data,axis=-1)
    data_std = np.std(data, axis=-1)
    data_tSNR = np.nan_to_num(data_mean / data_std)
    return data_tSNR


# calculate tSNR
for subid in sessidlist:  
    
    after_tSNR = np.zeros([np.size(vertex_number)])
    before_tSNR = np.zeros([np.size(vertex_number)])
    count = 0
	
    for runid in runidlist:
		
        before_path = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/before_data/sub-01_ses-1_task-motor_run-2_space-T1w_desc-preproc_bold_surface_lh.nii.gz'
        before_info = nib.load(before_path)
        before = before_info.get_fdata()[:,0,0,:]
        before_tSNR = before_tSNR + tSNR(before)   
		
        after_path = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/after_data/sub-01_ses-1_task-motor_run-2_space-T1w_desc-preproc_bold_denoised_surface_lh.nii.gz'
        after_info = nib.load(after_path)
        after = after_info.get_fdata()[:,0,0,:]
        after_tSNR = after_tSNR + tSNR(after)

        # print('++++++++++++++++++++++++++++before path++++++++++++++++++++++++++++')
        # print(before_path)
        # print('++++++++++++++++++++++++++++before info++++++++++++++++++++++++++++')
        # print(before_info)
        # print('++++++++++++++++++++++++++++before data++++++++++++++++++++++++++++')
        # print(before)
        print('++++++++++++++++++++++++++++before tSNR++++++++++++++++++++++++++++')
        print(before_tSNR)
        # print('++++++++++++++++++++++++++++after path++++++++++++++++++++++++++++')
        # print(after_path)
        # print('++++++++++++++++++++++++++++after info++++++++++++++++++++++++++++')
        # print(after_info)
        # print('++++++++++++++++++++++++++++after data++++++++++++++++++++++++++++')
        # print(after)
        print('++++++++++++++++++++++++++++after tSNR++++++++++++++++++++++++++++')
        print(after_tSNR)
        print('{0}_{1} done'.format(subid,runid))

    # after_tSNR = after_tSNR / len(runidlist)
    # before_tSNR = before_tSNR / len(runidlist)

    # # save pre-denoising tSNR image
    # img = nib.Nifti1Image(before_tSNR.reshape([vertex_number,1,1]), None, after_info.get_header())
    # result_name= '{0}_{1}_fsaverage_{2}.nii.gz'.format(smth_cdt, subid, hemi)
    # save_path = os.path.join(analysis_dir, result_name)
    # nib.save(img, save_path)
    #
    # # save post-denoising tSNR image
    # img = nib.Nifti1Image(after_tSNR.reshape([vertex_number,1,1]),
    #                       None, after_info.get_header())
    # result_name= '{0}_{1}_fsaverage_denoised_{2}.nii.gz'.format(
    #         smth_cdt, subid, hemi)
    # save_path = os.path.join(analysis_dir, result_name)
    # nib.save(img, save_path)
    #
    # print('Saving_{0}'.format(subid))

    before_counter = Counter(np.trunc(before_tSNR).tolist())
    for key in list(before_counter.keys()):
        if key < 1:
            del before_counter[key]

    after_counter = Counter(np.trunc(after_tSNR).tolist())
    for key in list(after_counter.keys()):
        if key < 1:
            del after_counter[key]
    # print(before_counter)
    # print(after_counter)

    def draw_from_dict(dicdata, RANGE, heng=0):
        by_value = sorted(dicdata.items(), key=lambda item: item[1], reverse=True)
        x = []
        y = []
        for d in by_value:
            x.append(d[0])
            y.append(d[1])
        if heng == 0:
            plt.bar(x[0:RANGE], y[0:RANGE])
            plt.show()
            return
        elif heng == 1:
            plt.barh(x[0:RANGE], y[0:RANGE])
            plt.show()
            return
        else:
            return "!"

    draw_from_dict(before_counter, 1000000)
  
print('======{0} done======='.format(subid))