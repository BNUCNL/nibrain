from mrfree.core.image import Image
import numpy as np
import nibabel as nib
# import pytest


def test_image():
    nii_data_path = '/nfs/s2/userhome/zhenzonglei/codebase/mrfree/mrfree/data/func_mni152.nii.gz'
    nii = nib.load(nii_data_path)
    img_a = Image(nii,'mni152')
    img_b = Image(nii,'mni152')
    img_c = img_a + img_b

    img_c.load(nii_data_path)

    print img_c.space, img_c.dims, img_c.voxsize

    roi = img_a.data[:,:,:,1] > 0.9
    ras_coords, crs_coords = img_a.get_roi_coords(roi)
    roi_data = img_a.get_roi_data(roi)

    print roi.shape, ras_coords.shape, crs_coords.shape, roi_data.shape


if __name__ == "__main__":
    test_image()
