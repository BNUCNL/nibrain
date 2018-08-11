
class Tractogram(object):
    """ Tractogram class represent fibers from fiber tracking

        Attributes
        ----------
        mesh: Mesh object, brain surface
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        dims: image dimensions, a 3x1 or 4x1 array
        """

    def __init__(self, tractogram=None, data=None, space=None):
        self.tractogram = tractogram
        self.data = data
        self.space = space

    def load_tractogram(self, filename):
        """ Load tractogram from a tractogram file, include tck, trk, vtk

        Parameters
        ----------
        filename: str
            Pathstr to a tractogram file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_tractogram(self, filename):
        """ Save tractogram to a tractogram file, include tck, trk, vtk

        Parameters
        ----------
        filename: str
            Pathstr to a tractogram file

        Returns
        -------
        self: a Lines object
        """
        pass

    def load_data(self, filename):
        """ Load tractogram scalar data from a tractogram scalar file
        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        self: a Surface object
        """
        pass

    def save_data(self, filename):
        """ Save tractogram scalar data from a tractogram scalar file
        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        bool: sucessful or not
        """
        pass
    
    
    def get_toi_data(self, toi=None):
        """ Get the data of the node within a toi

        Parameters
        ----------
        toi, a toi include the fiber id of interest
        if toi == None, return data from all vertices on the surface
        Returns
        -------
        values: NxT numpy array, scalar value from the toi
        """
        pass

    def get_toi_coords(self, toi=None):
        """ Get the coordinates of the node within a toi

        Parameters
        ----------
        toi, a toi include the fiber id of interest

        Returns
        -------
        coords: Nx3 numpy array, coords value from the toi
        """
        pass
