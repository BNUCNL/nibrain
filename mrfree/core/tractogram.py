
class Tractogram(object):
    """ Tractogram class represent fibers from fiber tracking

        Attributes
        ----------
        lines: streamline object from nibabel
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        dims: image dimensions, a 3x1 or 4x1 array
        """

    def __init__(self, lines=None, data=None, space=None):
        """

        Parameters
        ----------
        lines: a Lines object
        data: the scalar image data
        space: str, native,mni152
        """
        self.lines = lines
        self.data = data
        self.space = space

    def load_lines(self, filename):
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

    def save_lines(self, filename):
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
            Pathstr to a tractogram file

        Returns
        -------
        self: a Tractogram object
        """
        pass

    def save_data(self, filename):
        """ Save tractogram scalar data from a tractogram scalar file
        Parameters
        ----------
        filename: str
            Pathstr to a corresponding file

        Returns
        -------
        bool: sucessful or not
        """
        pass
    
    
    def get_toi_data(self, toi=None):
        """ Get the scalar data of the fiber within a toi

        Parameters
        ----------
        toi, a toi include the fiber id of interest
        if toi == None, return data from all vertices on the surface
        Returns
        -------
        data: NxT numpy array, scalar value from the toi
        """
        pass

    def get_toi_lines(self, toi=None):
        """ Get the coordinates of the node within a toi

        Parameters
        ----------
        toi, a toi include the fiber id of interest

        Returns
        -------
        lines: arraysequence, streamline from the toi
        """
        pass
