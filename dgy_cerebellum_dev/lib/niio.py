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

def get_surface_obj(data):
    sub_id = basic.rand_pick_sub()
    sub_32kdir = basic.get_32k_dir(sub_id)
    template = nibabel.load(os.path.join(sub_32kdir, f'{sub_id}_V1_MR.MyelinMap.32k_fs_LR.dscalar.nii'))
    header = template.header
    return nibabel.cifti2.cifti2.Cifti2Image(data, header = header)