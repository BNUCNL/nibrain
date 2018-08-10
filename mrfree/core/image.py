#!/usr/bin/env python

import numpy as np
import nibabel as nib


class Image(object):
    """ Image class represents brain image data from neuroimaging study.

          Attributes
          ----------
          image: nibabel image object
          space: a string, native, mni152
          itype: image type, a string.
          ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
          voxsize: voxel size, a 3x1 array
          dims: image dimensions, a 3x1 or 4x1 array
          """

    def __init__(self, image, space=None, itype=None):
        """
        Parameters
        ----------
        image: nibabel image object or a pathstr to a nibabel image file
        space: a string, native, mni152
        itype: image type, a string.
        ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
        voxsize: voxel size, a 3x1 array
        dims: image dimensions, a 3x1 or 4x1 array
        """

        self.image = image
        self.space = space
        self.itype = itype
        self.ras2vox = image.affine
        self.voxsize = image.header.pixdim
        self.dims = image.header.im

    @property
    def data(self):
        return self.image.get_fdata()

    @data.setter
    def data(self, data):
        assert data.ndim <= 4, "data should be 3d or 4d numpy array."
        self._data = data

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        possible_space = ('native','mni152')
        assert space in possible_space , "space should be in {0}".format(possible_space)
        self._space = space
        
    @property
    def itype(self):
        return self._itype

    @itype.setter
    def itype(self, itype):
        possible_itype = ('bold','dwi','anat')
        assert itype in possible_itype , "itype should be in {0}".format(possible_itype)
        self._itype = itype
        
    @property
    def ras2vox(self):
        return self._ras2vox

    @ras2vox.setter
    def ras2vox(self, ras2vox):
        assert ras2vox.shape == (4,4), "ras2vox should a 4x4 matrix."
        self._ras2vox = ras2vox
        
    @property
    def voxsize(self):
        return self._voxsize

    @voxsize.setter
    def voxsize(self, voxsize):
        assert voxsize.ndim == 1 and voxsize.shape[0] == 3, "voxsize should be 1 x 3 numpy array."
        self._voxsize = voxsize

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        assert dims.ndim == 1 and dims.shape[0] <= 4, "dims should be 1x3 or 1x4 numpy array."
        self._dims = dims

    def __add__(self, other):
        self.data = np.add(self.data, other.data)

    def __sub__(self, other):
        self.data = np.subtract(self.data, other.data)

    def __mul__(self, other):
        self.data = np.multiply(self.data, other.data)

    def __div__(self, other):
        self.data = np.divide(self.data, other.data)

    def get_coords(self, mask):
        """ Get the spatial coords of the voxels within the mask roi

        Parameters
        ----------
        mask

        Returns
        -------
        coords: Nx3 numpy array
        """
        pass


    def get_value(self, mask):
        """ Get the values of the voxels within the mask roi

        Parameters
        ----------
        mask

        Returns
        -------
        values: NxT numpy array, scalar value from the mask roi
        """
        pass

    def load(self, filename):
        """ Read image from a CIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: an Image obejct
        """
        pass

    def save(self, filename):
        """ Save the Image obejct to a CIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """

        pass
