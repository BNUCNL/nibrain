#!/usr/bin/env python

import numpy as np
from base import intersect2d,exclude2d
from image import Image

class Points(object):
    """Points represent a collection of spatial ponits
    
    Attributes
    ----------
    data:  Nx3 numpy array, points coordinates
    id: Nx1 numpy array, id for each point
    """
    def __init__(self, data=None, id=None, src=None):
        """
        Parameters
        ----------
        data:  Nx3 numpy array, points coordinates
        id: Nx1 numpy array, id for each point
        src: str, source of the points data
        """
        self.data  = data
        self.id = id

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert data.ndim == 2 and data.shape[1] == 3, "data should be N x 3 np array."
        self._data = data

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    def merge(self, other):
        """ Merge other Points object into self
        
        Parameters
        ----------
        other: Points object to be merged

        Returns
        -------
        self: the merged object

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = np.vstack(self.data)
        self.data = np.unique(self.data,axis=0)
        return self

    def intersect(self,other):
        """ Intersect with other Points object

        Parameters
        ----------
        other: Points object to be intersectd with this object

        Returns
        -------
        self: the intersected object

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = intersect2d(self.data, other.data)
        return self

    def exclude(self, other):
        """ Exclude other Points object from this object

        Parameters
        ----------
        other: Points object to be merged

        Returns
        -------
        self: the object after excluding others

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = exclude2d(self.data, other.data)
        return self

    def get_center(self):
        """ Get the center of the points set
        
        Parameters
        ----------

        Returns
        -------
        center: 1x3 numpy array, the center coordinates

        """
        
        return np.mean(self.data,axis=0)

    def update_from_image(self, image):
        """ Construct Scalar object by reading a CIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: a Lines object
        """
        # use Image object to do the work
        if ~isinstance(image, Image):
            image = Image(image)

        self.data = image.get_roi_coords()


class Lines(object):
    def __init__(self, data, id, source=None):
        """
        Parameters
        ----------
        data: geometry data, a sequence of array.
        id: the id for each array.
        source: source of the geometry data, a string.
        """
        self.data = data
        self.id = id
        self.src = source

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self,src):
        self._src = src

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    def merge(self, other):
        """ Merge other tract into the Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another lines
        axis: integer, 0 or 1

        Return
        ----------
        self: merged Lines
        """
        assert isinstance(other, Lines), "other should be a Lines object"
        pass

    def intersect(self,other):
        """ Intersect with other Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another Lines 

        Return
        ----------
        self with intersection from two Lines
        """
        pass

    def exclude(self, other):
        """ Exclude other Lines from the current Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another Lines 

        Return
        ----------
        self:  The Lines after excluding other Lines
        """
        pass

    def equidistant_resample(self, num_segment):
        """ Resample the Lines with equidistantance
        
        Parameters
        ----------
        num_segment: int, number of segment to be sampled

        Returns
        -------
        
        self: a resampled Lines

        """
        pass

    def skeleton(self):
        """Find the skeletion of the line set
        
        Returns
        -------
        skeleton: a Line object

        """
        pass


class Mesh(object):
    """Mesh class represents geometry mesh
    
        Attributes
        ----------
        vertices
        faces

    
    """
    
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def __eq__(self, other):
        return  np.array_equal(self.vertices, self.vertices)

    def update_from_freesurfer(self):
        pass 

    def update_from_gifti(self, filename):
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