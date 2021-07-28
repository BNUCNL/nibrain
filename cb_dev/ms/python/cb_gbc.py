#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 02:12:55 2021

@author: masai

Brain structures list:   'CIFTI_STRUCTURE_CORTEX_LEFT',
                         'CIFTI_STRUCTURE_CORTEX_RIGHT',
                         'CIFTI_STRUCTURE_ACCUMBENS_LEFT',
                         'CIFTI_STRUCTURE_ACCUMBENS_RIGHT',
                         'CIFTI_STRUCTURE_AMYGDALA_LEFT',
                         'CIFTI_STRUCTURE_AMYGDALA_RIGHT',
                         'CIFTI_STRUCTURE_BRAIN_STEM',
                         'CIFTI_STRUCTURE_CAUDATE_LEFT',
                         'CIFTI_STRUCTURE_CAUDATE_RIGHT',
                         'CIFTI_STRUCTURE_CEREBELLUM_LEFT',
                         'CIFTI_STRUCTURE_CEREBELLUM_RIGHT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT',
                         'CIFTI_STRUCTURE_PALLIDUM_LEFT',
                         'CIFTI_STRUCTURE_PALLIDUM_RIGHT',
                         'CIFTI_STRUCTURE_PUTAMEN_LEFT',
                         'CIFTI_STRUCTURE_PUTAMEN_RIGHT',
                         'CIFTI_STRUCTURE_THALAMUS_LEFT',
                         'CIFTI_STRUCTURE_THALAMUS_RIGHT'
"""

# %%
import os
import numpy as np
from scipy.spatial.distance import cdist
from ioTools import CiftiReader, save2nifti

# %%
# test path
resting_file = '/nfs/e1/HCPD/fmriresults01/HCD0008117_V1_MR/MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii'
net_roi_L_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/SeparateHemispheres/CortexSubcortex_ColeAnticevic_NetPartition_netassignments_v1_L.dlabel.nii'
net_roi_R_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/SeparateHemispheres/CortexSubcortex_ColeAnticevic_NetPartition_netassignments_v1_R.dlabel.nii'

# %%
# prepare data matrix
# load resting state image
resting_img = CiftiReader(resting_file)

# get offsets & count of all brain structures
brain_structures = resting_img.brain_structures
offset_index = {}
for structure_name in brain_structures:
    offset_index[structure_name] = resting_img.brain_models([structure_name])[0].index_offset
count_index = {}
for structure_name in brain_structures:
    count_index[structure_name] = resting_img.brain_models([structure_name])[0].index_count

# get whole brain & cerebellum & roi data
whole_brain_data = resting_img.get_data(structure=None)  
cb_data = whole_brain_data[:, offset_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT']:offset_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT']+count_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT']+count_index['CIFTI_STRUCTURE_CEREBELLUM_RIGHT']]

# %%
# caculate connectivity matrix and save to nifti file
cb_brain_conn = 1 - cdist(cb_data.T, whole_brain_data.T, metric='correlation')

# %%
# connectivity matrix information dicts
subcortex_index = {}
for hemisphere in ['LEFT', 'RIGHT']:
    subcortex_index_hemi = tuple()
    for subcortex_structure_name in brain_structures[2:6] + brain_structures[7:9] + brain_structures[11:]:
        if hemisphere in subcortex_structure_name:
            subcortex_index_hemi += tuple(np.arange(offset_index[subcortex_structure_name], offset_index[subcortex_structure_name]+count_index[subcortex_structure_name]))
    subcortex_index['SUBCORTEX_' + hemisphere] = subcortex_index_hemi
    
net_roi_L_index = []
net_roi_R_index = []
net_L_data = CiftiReader(net_roi_L_file).get_data(structure=None)
net_R_data = CiftiReader(net_roi_R_file).get_data(structure=None)
for net_id in np.arange(1,13): # ROI id ranges from 1 to 12
    net_roi_L_index.append(tuple(np.where(net_L_data==net_id)[1]))
    net_roi_R_index.append(tuple(np.where(net_R_data==net_id)[1]))
    
structure_index = {'cb_cortex_L_conn' : [tuple(np.arange(offset_index['CIFTI_STRUCTURE_CORTEX_LEFT'], offset_index['CIFTI_STRUCTURE_CORTEX_RIGHT']))],
                   'cb_cortex_R_conn' : [tuple(np.arange(offset_index['CIFTI_STRUCTURE_CORTEX_RIGHT'], offset_index['CIFTI_STRUCTURE_ACCUMBENS_LEFT']))],
                   'cb_subcortex_L_conn' : [subcortex_index['SUBCORTEX_LEFT']],
                   'cb_subcortex_R_conn' : [subcortex_index['SUBCORTEX_RIGHT']],
                   'cb_cb_L_conn' : [tuple(np.arange(offset_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT'], offset_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT']+count_index['CIFTI_STRUCTURE_CEREBELLUM_LEFT']))],
                   'cb_cb_R_conn' : [tuple(np.arange(offset_index['CIFTI_STRUCTURE_CEREBELLUM_RIGHT'], offset_index['CIFTI_STRUCTURE_CEREBELLUM_RIGHT']+count_index['CIFTI_STRUCTURE_CEREBELLUM_RIGHT']))],
                   'cb_net_L_conn' : net_roi_L_index,
                   'cb_net_R_conn' : net_roi_R_index,
                   'cb_brainstem_conn' : [tuple(np.arange(offset_index['CIFTI_STRUCTURE_BRAIN_STEM'], offset_index['CIFTI_STRUCTURE_BRAIN_STEM']+count_index['CIFTI_STRUCTURE_BRAIN_STEM']))]
                   }
save_shape = {'cb_cortex_L_conn' : (91,109,91,1),
              'cb_cortex_R_conn' : (91,109,91,1),
              'cb_subcortex_L_conn' : (91,109,91,1),
              'cb_subcortex_R_conn' : (91,109,91,1),
              'cb_cb_L_conn' : (91,109,91,1),
              'cb_cb_R_conn' : (91,109,91,1),
              'cb_net_L_conn' : (91,109,91,12),
              'cb_net_R_conn' : (91,109,91,12),
              'cb_brainstem_conn' : (91,109,91,1),
              } # 91x109x91 is shape of 3D nifti matrix from MNI template and 12 is ROIs number of Cole Anticevic net

# %%
# get affine and cerebellum index
affine_matrix = resting_img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
cb_L_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT', zeroize=False)[2]
cb_R_index = resting_img.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_RIGHT', zeroize=False)[2]
cb_L_index.extend(cb_R_index)
cb_index_matrix = np.asarray(cb_L_index)

# caculate mean of connectivity between cerebellum and other brain structure & Cole Anticevic net
for name, index in structure_index.items():
    conn_save_matrix = np.zeros(save_shape[name])
    for roi_id in np.arange(len(structure_index)):
        conn_mean_matrix = np.mean(cb_brain_conn[:, index[roi_id]], axis=1)
        conn_save_matrix[cb_index_matrix[:,0],cb_index_matrix[:,1],cb_index_matrix[:,2],roi_id] = conn_mean_matrix[:,roi_id]
    np.squeeze(conn_save_matrix)
    save2nifti(os.path.join('/nfs/e2/workingshop/masai/code/cb/python', name+'.nii.gz'), conn_save_matrix, affine=affine_matrix, header=None)
