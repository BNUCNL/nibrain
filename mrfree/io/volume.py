
import nibabel as nib


def load_nifti(image_name, mask_name):
    """ Load nifti data within a mask

    Parameters
    ----------
    image_name: pathstr for target image
    mask_name: pathstr for mask to provide the spatial constrain

    Returns
    -------
    image: nifti
    coords: Nx3 numpy array, spatial coordinates
    value : NxT numpy array, scalar feature value
    """
    pass
