from mrfree.core.image import Image
import numpy as np
import nibabel as nib
# import pytest

nii_data_path = '/nfs/e5/stanford/ABA/brainmap/cytoAtlas/cytoPM.nii.gz'
nii = nib.load(nii_data_path)