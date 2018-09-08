#!/usr/bin/env python

import numpy as np
from image import Image
from surface import Surface
from tractogram import Tractogram

class Points(object):
    """Points represent a collection of spatial ponits
    
    Attributes
    ----------
    coords:  Nx3 numpy array, points coordinates
    id: Nx1 numpy array,tuple or list, id for each point
    src: source image or surface obejct which the coords were dervied
    """
    def __init__(self, coords, id=None):
        """
        Parameters
        ----------
        coords:  Nx3 numpy array, points coordinates
        id: Nx1 numpy array, id for each point
        src: source image or surface obejct which the coords were dervied
        """

        self.coords  = coords
        if id is None:
            id = range(coords.shape[0])
        elif np.asarray(id).shape[0] != coords.shape[0]:
            raise ValueError("id length is not equal to the length of the coords")

        self.id = id

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        assert coords.ndim == 2 and coords.shape[1] == 3, "coords should be N x 3 np array."
        self._coords = coords

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
        self.coords = np.vstack((self.coords, other.coords))
        self.id = np.vstack((self.id, other.id))
        self.id, idx = np.unique(self.id, return_index=True)
        self.coords = self.coords[idx,:]
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
        idx = np.in1d(self.id, other.id)
        self.id = self.id[idx]
        self.coords = self.coords[idx,:]
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

        idx = np.logical_not(np.in1d(self.id, other.id))
        self.id = self.id[idx]
        self.coords = self.coords[idx,:]
        return self

    def get_center(self):
        """ Get the center of the points set
        
        Parameters
        ----------

        Returns
        -------
        center: 1x3 numpy array, the center coordinates

        """
        
        return np.mean(self.coords,axis=0)
    
class Lines(object):
    def __init__(self, coords, id=None, src=None):
        """
        Parameters
        ----------
        coords: geometry coords, a sequence of array.
        id: the id for each array.
        src: source image or surface obejct which the coords were dervied
        """
        self.coords = coords
        if id is None:
            id = range(len(coords))
        elif np.asarray(id).shape[0] != len(coords):
            raise ValueError("id length is not equal to the length of the coords")

        self.id = id

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = coords

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        assert isinstance(src, Tractogram), "src should a Tractogram object."
        self._src = src

    def merge(self, other):
        """ Merge other Lines into the Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another lines
        axis: integer, 0 or 1

        Return
        ----------
        self: merged Lines
        """
        assert isinstance(other, Lines), "other should be a Lines object"
        self.coords = np.vstack((self.coords, other.coords))
        self.id = np.vstack((self.id, other.id))
        self.id, idx = np.unique(self.id, return_index=True)
        self.coords = self.coords[idx, :]
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
        idx = np.in1d(self.id, other.id)
        self.id = self.id(idx)
        del self.coords[np.nonzero(np.logical_not(idx))]

    def exclude(self, other):
        """ Exclude other Lines from the current Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another Lines 

        Return
        ----------
        self:  The Lines after excluding other Lines
        """
        idx = np.in1d(self.id, other.id)
        self.id = self.id(np.logical_not(idx))
        del self.coords[np.nonzero(idx)]

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

    def save(self, filename):
        self.src.save_lines(filename)


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