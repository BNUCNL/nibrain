import os
import json
import nibabel
import numpy as np
from typing import Union
from collections import Iterable

from . import basic

class BaseROI(object):
    """Base class of ROI defining most basic interfaces.
    """
    roi_map = None # `nibabel` imaging object
    name_id_mapping = {} # dict, of format as {roi_name: roi_id}
    
    def get_data(self):
        raise NotImplementedError
    
    def get_idx(self, roi_id: Union[int, Iterable], exclude_specified_id: bool = False) -> np.ndarray:
        """For an ROI index, returns corresponding indices.

        Args:
            roi_id (Union[int, Iterable]): ROI index. If being an iterable, indices of conjuncted regions will be returned(OR).
            exclude_specified_id (bool, optional): If set to `True`, indices except given ROI will be returned(NOT). Defaults to False.

        Returns:
            np.ndarray: Indices of given ROI.
        """
        if not self.roi_map:
            raise NotImplementedError
        
        if isinstance(roi_id, Iterable):
            indices = np.ones_like(self.get_data())
            for i in roi_id:
                indices *= self.get_data() - i
            return np.where((indices != 0) if exclude_specified_id else (indices == 0))
        else:
            return (self.get_data() != roi_id) if exclude_specified_id \
                else np.where(self.get_data() == roi_id)

    def get_roi_id(self, roi_name: str) -> Union[int, None, np.nan]:
        """Returns ROI index of given ROI name if there is a corresponding record in `name_id_mapping`.

        Args:
            roi_name (str): ROI name, ususally in upper case as 'CEREBELLUM'.

        Returns:
            Union[int, None, np.nan]: ROI index.
        """
        if roi_name in self.name_id_mapping:
            return self.name_id_mapping[roi_name]
        else:
            raise NotImplementedError
    
    def invert(self, data: np.ndarray, roi_id: Union[int, Iterable], exclude_specified_id: bool = False, fill_val: float = np.nan) -> np.ndarray:
        """Returns array in original space with data of given ROI.

        Args:
            data (np.ndarray): Data within given ROI.
            roi_id (Union[int, Iterable]): ROI index. If being an iterable, indices of conjuncted regions will be returned(OR).
            exclude_specified_id (bool, optional): If set to `True`, indices except given ROI will be returned(NOT). Defaults to False.
            fill_val (float, optional): Background value of original space except given ROI. Defaults to np.nan.

        Returns:
            np.ndarray: Array in original space.
        """
        output = np.ones_like(self.get_data()) * fill_val
        output[self.get_idx(roi_id, exclude_specified_id)] = data
        return output
    
    def __call__(self, roi_id, exclude_specified_id=False):
        return self.get_idx(roi_id, exclude_specified_id)

    def __iter__(self):
        return iter(self.name_id_mapping.items())

class SurfROI(BaseROI):
    def get_data(self):
        return self.roi_map.get_fdata()[0]

class MNIVolumeROI(BaseROI):
    def get_data(self):
        return self.roi_map.get_fdata()

class SurfMMP(SurfROI):
    roi_map = nibabel.load(os.path.join(basic.HCP_AVERAGE_DATADIR,
        'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'))

class VolumeWMParc(MNIVolumeROI):
    roi_map = nibabel.load(os.path.join(basic.DATA_DIR(), 'ROI', 'Atlas_wmparc.2.nii.gz'))
    name_id_mapping = json.loads(open(os.path.join(basic.DATA_DIR(), 'ROI', 'wmparc_mapping.json')).read())

class VolumeAtlas(MNIVolumeROI):
    roi_map = nibabel.load(os.path.join(basic.DATA_DIR(), 'ROI', 'Atlas_ROIs.2.nii'))
    name_id_mapping = json.loads(open(os.path.join(basic.DATA_DIR(), 'ROI', 'Atlas_ROIs_mapping.json')).read())

    def get_sc(self):
        roi_id = (*self.get_roi_id('CEREBELLUM'), 0)
        indices = self.get_idx(roi_id, True)
        return indices

class VolumeMDTB(MNIVolumeROI):
    roi_map = nibabel.load(os.path.join(basic.DATA_DIR(), 'ROI', 'mdtb_mni.2.nii.gz'))
    name_id_mapping = json.loads(open(os.path.join(basic.DATA_DIR(), 'ROI', 'mdtb_mapping.json')).read())