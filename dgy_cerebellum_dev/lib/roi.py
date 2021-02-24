import os
import re
import nibabel
import collections
import numpy as np
from numpy.core.numeric import indices
import pandas as pd

from . import basic

#--------------------------
# Surface ROIs given by MMP

MMP_RAW = nibabel.load(os.path.join(basic.HCP_AVERAGE_DATADIR,
    'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'))

def get_MMP_32k_indices(roi_id):    
    return np.where(MMP_RAW.get_fdata()[0] == roi_id)[0]

#------------
# Volume ROIs

class _VolumeROI(object):
    '''Volume ROI indices in .2 resolution.

    Further instructions:

    - ROI_DF: Data frame read from 'ROI.csv'

    - NAME_TO_INDEX: Transformed from ROI_DF as {roi_name: index}

    - CEREBELLUM, BRAINSTEM: Corresponding ROI indices

    - NAMES: ROI names without LEFT/RIGHT suffixes

    - NAMES_TO_INDEX_LR: Transformed from NAME_TO_INDEX as {roi_name: (index_l, index_r)}

    - ROI_RAW: 3-D array containing ROI indices of each volume, read from a random subject directory

    '''
    ROI_DF = pd.read_csv(os.path.join(basic.DATA_DIR(), 'ROIs.csv'))
    NAME_TO_INDEX = {key: value for key, value in zip(list(ROI_DF['roi']), list(ROI_DF['index']))}
    
    CEREBELLUM = (8, 47)
    BRAINSTEM = 16

    def __init__(self):
        self.NAMES = set(map(
            lambda s: re.match(r'([A-Z]+(_[A-Z]+)?)_(RIGHT|LEFT)', s).groups()[0], list(self.NAME_TO_INDEX.keys())
        ))
        self.NAMES_TO_INDEX_LR = {key: (self.NAME_TO_INDEX[f'{key}_LEFT'], self.NAME_TO_INDEX[f'{key}_RIGHT']) for key in self.NAMES}
        
        sub_dir = basic.get_mni_dir(basic.rand_pick_sub())
        self.ROI_ARR = nibabel.load(os.path.join(sub_dir, 'ROIs', 'Atlas_ROIs.2.nii.gz')).get_fdata()

VOLUME_ROI = _VolumeROI()

def get_volume_indices(roi_id, include=True):
    """For the given ROI index/indices, return corresponding positions according to 'Atlas_ROIs.2.nii.gz'.

    Args:
        roi_id (int, iterable): ROI index or indices.
        include (bool, optional): If set to False, indices of volumes not belonging to given ROI will be returned.
    """
    
    if isinstance(roi_id, collections.Iterable):
        indices = np.ones(VOLUME_ROI.ROI_ARR.shape)
        for i in roi_id:
            indices *= VOLUME_ROI.ROI_ARR - i
        return np.where((indices == 0) if include else (indices != 0))
    else:
        return np.where((VOLUME_ROI.ROI_ARR == roi_id) if include else (VOLUME_ROI.ROI_ARR != roi_id))