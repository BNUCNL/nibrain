import os
import nibabel

from . import basic

def get_volume_obj(data):
    """With given data, generate a corresponding nibabel object to enable file I/O.
    """
    sub_id = basic.rand_pick_sub()
    sub_dir = basic.get_mni_dir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    return nibabel.nifti1.Nifti1Image(data, t1.affine)