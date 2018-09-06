from nibabel.streamlines import tck
from nibabel import trackvis

class Tractography(object):
    """ Class Tractography represents the result of fiber tracking process.

        Attributes
        ----------
        lines: streamline object from nibabel
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        dims: image dimensions, a 3x1 or 4x1 array
        """

    def __init__(self, tractography=None, data=None, space=None):
        """
        Parameters
        ----------
        lines: a Lines object
        data: the scalar image data
        space: str, native, mni152
        """
        self.tractography = tractography
        self.data = data
        self.space = space

    def load_lines(self, filename):
        """ Load tractography from a tractography file, include tck, trk, vtk(tck file will be accepted  temporarily  )

        Parameters
        ----------
        filename: str
            Pathstr to a tractography file

        Returns
        -------
        self: a Lines object
        """
        self.tractography = tck.TckFile.load(filename)

    def save_tck(self,save_form,save_param,out_path=None):
        """
        save streamlines data
        Parameters
        ----------
        save_form: str, select one file format to save,include tck, trk, vtk
        streamlines: iterable of ndarrays or :class:`ArraySequence`, optional
                Sequence of $T$ streamlines. Each streamline is an ndarray of
                shape ($N_t$, 3) where $N_t$ is the number of points of
                streamline $t$.
        header:streamlines data header
        data_per_streamline: dict of iterable of ndarrays, optional
                Dictionary where the items are (str, iterable).
                Each key represents an information $i$ to be kept alongside every
                streamline, and its associated value is an iterable of ndarrays of
                shape ($P_i$,) where $P_i$ is the number of scalar values to store
                for that particular information $i$.
        data_per_point: dict of iterable of ndarrays, optional
                Dictionary where the items are (str, iterable).
                Each key represents an information $i$ to be kept alongside every
                point of every streamline, and its associated value is an iterable
                of ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of
                points for a particular streamline $t$ and $M_i$ is the number
                scalar values to store for that particular information $i$.
        affine_to_rasmm: ndarray of shape (4, 4) or None, optional
                Transformation matrix that brings the streamlines contained in
                this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
                refers to the center of the voxel. By default, the streamlines
                are in an unknown space, i.e. affine_to_rasmm is None.
        out_path: filename for saving
        Return
        ------
        streamlines data
        """
        if save_form == 'tck':
            streamline = save_param[0]
            data_per_streamline = save_param[1]
            data_per_point = save_param[2]
            affine_to_rasmm = save_param[3]
            tractogram = streamlines.tractogram.Tractogram(streamlines=streamline,data_per_streamline=data_per_streamline,
                                                        data_per_point=data_per_point,affine_to_rasmm=affine_to_rasmm)
            datdat = nibtck.TckFile(tractogram=tractogram, header=header)
            datdat.save(out_path)

    def load_data(self, tractography=None):
        """ Load fiber streamlines data from a tractography file
        Parameters
        ----------
        tractography: str of filepath or line object

        """
        if tractography == None:
            self.data = self.lines.streamlines
        else:
            self.tractography = tck.TckFile.load(tractography)
            self.data = self.tractography.streamlines

    def save_data(self, filename):
        """ Save tractogram scalar data to a tractogram scalar file
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
        """Get the coordinates of the node within a toi

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
