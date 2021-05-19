import os
import nibabel
import numpy as np

from . import basic

def get_vol_obj(data: np.ndarray) -> nibabel.nifti1.Nifti1Image:
    """With given volume data, generates a corresponding nibabel object to enable file I/O.

    Note that a default affine of .2 resolution is used in this function.
    
    Args:
        data (np.ndarray): Array containing original or derived neuroimaging data.

    Returns:
        nibabel.nifti1.Nifti1Image: `nibabel` object containing given data.
    """
    sub_id = basic.rand_pick_sub()
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    return nibabel.nifti1.Nifti1Image(data, t1.affine)

def get_surf_obj(data: np.ndarray) -> nibabel.cifti2.cifti2.Cifti2Image:
    """With given surface data, generates a corresponding nibabel object to enable file I/O.

    Note that a default header and affine of .2 resolution and `fsaverage32k` is used in this function.
    
    Args:
        data (np.ndarray): Array containing original or derived neuroimaging data.

    Returns:
        nibabel.cifti2.cifti2.Cifti2Image: `nibabel` object containing given data.
    """
    sub_id = basic.rand_pick_sub()
    sub_32kdir = basic.get_32k_dir(sub_id)
    template = nibabel.load(os.path.join(sub_32kdir, f'{sub_id}_V1_MR.MyelinMap.32k_fs_LR.dscalar.nii'))
    header = template.header
    return nibabel.cifti2.cifti2.Cifti2Image(data, header = header)