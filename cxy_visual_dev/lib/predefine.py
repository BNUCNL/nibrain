import nibabel as nib
from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName


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
            self.n_roi = len(self.roi2label)
        else:
            raise ValueError(f'{atlas_name} is not supported at present!')
