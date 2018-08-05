
class surface(object):
    """ surface was used to represent brain surface data from surface analysis
        Attributes
        ----------
        surface: triangular patch based surface geometry
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
        dims: image dimensions, a 3x1 or 4x1 array
        src: source of the image data, a string.
        """

    def __init__(self):
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

    def read_surf(self, filename):
        pass


    def read_scalar(self, filename):
        """ Read surface geometry from a GIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------
        self: an surface obejct
        """
        pass

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
