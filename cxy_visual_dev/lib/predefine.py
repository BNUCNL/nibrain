import numpy as np
import nibabel as nib
from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName


LR_count_32k = 59412
L_offset_32k = 0
L_count_32k = 29696
R_offset_32k = 29696
R_count_32k = 29716

mmp_file = '/nfs/p1/atlases/multimodal_glasser/surface/'\
           'MMP_mpmLR32k.dlabel.nii'


class Atlas:
    """
    atlas_name: atlas name
    maps: (n_map, n_vtx) numpy array
    roi2label: key - ROI name; value - ROI label
    """

    def __init__(self, atlas_name=None):
        """
        Args:
            atlas_name (str, optional): atlas name.
                Defaults to None.
        """
        if atlas_name is None:
            self.atlas_name = None
            self.maps = None
            self.roi2label = None
            self.n_roi = None
        else:
            self.set(atlas_name)

    def set(self, atlas_name):
        """
        Set atlas

        Args:
            atlas_name (str): atlas name
                'LR': 左右脑分别作为两个大ROI
                'Cole_visual_LR': ColeNet的左右视觉相关区域分别作为两个大ROI
                'Cole_visual_ROI': ColeNet和视觉相关的各个ROI
        """
        self.atlas_name = atlas_name

        if atlas_name == 'Cole_visual_ROI':
            self.maps = nib.load(mmp_file).get_fdata()
            cole_names = ['Primary Visual', 'Secondary Visual',
                          'Posterior Multimodal', 'Ventral Multimodal']
            self.roi2label = get_parcel2label_by_ColeName(cole_names)
        elif atlas_name == 'LR':
            self.maps = np.ones((1, LR_count_32k), dtype=np.uint8)
            self.maps[0, R_offset_32k:(R_offset_32k+R_count_32k)] = 2
            self.roi2label = {'L_cortex': 1, 'R_cortex': 2}
        elif atlas_name == 'Cole_visual_LR':
            mmp_map = nib.load(mmp_file).get_fdata()
            self.maps = np.zeros_like(mmp_map, dtype=np.uint8)
            cole_names = ['Primary Visual', 'Secondary Visual',
                          'Posterior Multimodal', 'Ventral Multimodal']
            parcel2label = get_parcel2label_by_ColeName(cole_names)
            for roi, lbl in parcel2label.items():
                if roi.startswith('L_'):
                    self.maps[mmp_map == lbl] = 1
                elif roi.startswith('R_'):
                    self.maps[mmp_map == lbl] = 2
                else:
                    raise ValueError('parcel name must start with L_ or R_!')
            self.roi2label = {'L_cole_visual': 1, 'R_cole_visual': 2}
        else:
            raise ValueError(f'{atlas_name} is not supported at present!')
        self.n_roi = len(self.roi2label)
