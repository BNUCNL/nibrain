#!/usr/bin/env python

import numpy as np
import nibabel as nib


class Image(object):
    def __init__(self, data, space, itype, ras2vox, voxsize, dims, src=None):
        """ Image class was designed to represent brain image data from neuroimaging study.

              Parameters
              ----------
              data: image data, a 3d or 4d array
              space: a string, native, mni152
              itype: image type, a string.
              ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
              voxsize: voxel size, a 3x1 array
              dims: image dimensions, a 3x1 or 4x1 array
              src: source of the image data, a string.
              """

        self.data = data
        self.space = space
        self.itype = itype
        self.ras2vox = ras2vox
        self.voxsize = voxsize
        self.dims = dims
        self.src = src 

    @property
    def data(self):
        return self._data

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

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        assert isinstance(src,basestring), "src should be a string."
        self._src = src

    def __add__(self, other):
        self.data = np.add(self.data, other.data)

    def __sub__(self, other):
        self.data = np.subtract(self.data, other.data)

    def __mul__(self, other):
        self.data = np.multiply(self.data, other.data)

    def __div__(self, other):
        self.data = np.divide(self.data, other.data)

    def read_from_cifti(self, filename):
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

    def save_to_cifti(self, filename):
        """ Save the Image obejct to a CIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """

        pass

    def read_from_nifti(self, filename):
        """ Read image from a NIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a NIFTI file

        Returns
        -------
        self: an Image obejct
        """

        pass

    def save_to_nifti(self, filename):
        """ Save the Image obejct to a NIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a NIFTI file

        Returns
        -------

        """
        pass

    def read_from_gifti(self, filename):
        """ Read image from a NIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: an Image obejct
        """
        pass

    def save_to_gifti(self, filename):
        """ Save the Image obejct to a GIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------

        """
        pass
