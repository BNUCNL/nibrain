import os
import numpy as np
import nibabel as nib
from nibabel.spatialimages import ImageFileError


class Image(object):
    """ Image class represents brain image data from neuroimaging study.

          Attributes
          ----------
          image: nibabel image object
          space: a string, native, mni152
          itype: image type, a string.
          ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
          voxsize: voxel size, a tuple
          dims: image dimensions, a tuple
          """

    def __init__(self, image, space=None):
        """
        Parameters
        ----------
        image: nibabel image object or a pathstr to a nibabel image file
        space: a string, native, mni152
        vox2ras: transform matrix from voxel coords to ras coords , a 4x4 array
        voxsize: voxel size, a tuple
        dims: image dimensions, a tuple
        """
        #if isinstance(image, basestring):
        #    self.image = nib.load(image)
        #elif isinstance(image, nib.Nifti1Image):
        #    self.image = image

        self.image = image
        self.data = image.get_data()
        self.vox2ras = image.affine
        self.voxsize = image.header.get_zooms()
        self.dims = image.shape
        self.space = space

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if isinstance(image, basestring):
            self._image = nib.load(image)
        elif isinstance(image, nib.Nifti1Image):
            self._image = image

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
    def vox2ras(self):
        return self._vox2ras

    @vox2ras.setter
    def vox2ras(self, vox2ras):
        assert vox2ras.shape == (4,4), "vox2ras should a 4x4 matrix."
        self._vox2ras = vox2ras
        
    @property
    def voxsize(self):
        return self._voxsize

    @voxsize.setter
    def voxsize(self, voxsize):
        assert len(voxsize) <= 4, "voxsize should be a tuple with length less than 4."
        self._voxsize = voxsize

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dims):
        assert len(dims) <= 4, "dims should be a tuple with length less than 4."
        self._dims = dims

    def update(self):
        """ Update other attributes according to the image attributes

        Returns
        -------
        None
        """
        self.data = self.image.get_data()
        self.vox2ras = self.image.affine
        self.voxsize = self.image.header.get_zooms()
        self.dims = self.image.shape

    def __add__(self, other):
        self.data = np.add(self.data, other.data)
        return self

    def __sub__(self, other):
        self.data = np.subtract(self.data, other.data)
        return self

    def __mul__(self, other):
        self.data = np.multiply(self.data, other.data)
        return self

    def __div__(self, other):
        self.data = np.divide(self.data, other.data)
        return self

    def get_roi_coords(self, roi=None):
        """ Get the spatial coords of the voxels within a roi

        Parameters
        ----------
        roi: nibabel image object or numpy array with the same shape as image

        Returns
        -------
        coords: Nx3 numpy array
        """
        if roi is None:
            roi = self.data[:, :, :, 0]
        elif isinstance(roi, nib.Nifti1Image):
            roi = roi.get_data()

        if isinstance(roi, np.ndarray) and roi.shape == self.dims[0:3]:
            crs_coords = np.asarray(np.nonzero(roi))
        else:
            raise ValueError("Passed roi is not of the right shape")

        crs_coords = np.vstack((crs_coords, np.ones((1,crs_coords.shape[1]))))
        ras_coords = np.matmul(self.vox2ras, crs_coords)
        ras_coords = ras_coords[0:3,:]
        crs_coords = crs_coords[0:3,:].astype(int)

        return ras_coords.T, crs_coords.T

    def get_roi_data(self, roi=None):
        """ Get the data of the voxels within a roi

        Parameters
        ----------
        roi: nibabel image object or numpy array with the same shape as image


        Returns
        -------
        data: NxT numpy array, scalar value from the mask roi
        """
        _, crs_coords = self.get_roi_coords(roi)
        # crs_coords = crs_coords.astype(int)
        data = self.data[crs_coords[:,0], crs_coords[:,1], crs_coords[:,2],:]

        return data

    def load(self, filename):
        """ Read image from a image file include nifti and cifti

        Parameters
        ----------
        filename: str
            Pathstr to a image file

        Returns
        -------
        self: an Image obejct
        """

        if (filename.endswith('.nii.gz')) or (filename.endswith('.nii') and filename.count('.') == 1):
            self.image = nib.load(filename)
            self.update()

        else:
            suffix = os.path.split(filename)[1].split('.')[-1]
            raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    def save(self, filename):
        """ Save the image to a image file

        Parameters
        ----------
        filename: str
            Pathstr to a image file

        Returns
        -------

        """

        nib.save(self.image, filename)
