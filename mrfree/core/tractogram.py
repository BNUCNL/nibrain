
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
        self.tractogram  = tractogram
        self.data = data
        self.space = space

    def load_tractogram(self, filename):
        """ Construct Lines object by reading a TCK file

        Parameters
        ----------
        filename: str
            Pathstr to a TCK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_tractogram(self, filename):
        """ Save Lines object to a TCK file

        Parameters
        ----------
        filename: str
            Pathstr to a TCK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def load_data(self, filename):
        """ Construct Lines object by reading a TRK file

        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_data(self, filename):
        """ Save Lines object to a TRK file

        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        self: a Lines object
        """
        pass