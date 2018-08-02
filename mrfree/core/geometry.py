#!/usr/bin/env python

# attributes class

import numpy as np
from mrfree.algorithms.geometry.geo_tools import faces_to_edges, faces_to_adjmatrix


class Geometry(object):
    def __init__(self, name, coords=None, faces=None, index=None):
        """
        Init Geometry.

        Parameters
        ----------
        name: the name of where geometry indicated, like 'inflated', 'sphere' etc.
        coords: coords of vertexes, should be N*3 array.
        faces: faces of vertexes, should be M*3 array.
        index: vertexes index in this geometry, should be K*1 array.
        """
        self._surface_type = ['white', 'pial', 'inflated', 'sphere']

        self.name = name
        self.coords = coords
        self.faces = faces
        self.index = index

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), "Input 'name' should be string."
        assert name in self._surface_type, "Name should be in {0}".format(self._surface_type)
        self._name = name

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        assert coords.ndim == 2, "Input should be 2-dim."
        assert coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._coords = coords

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        assert faces.ndim == 2, "Input should be 2-dim."
        assert faces.shape[1] == 3, "The shape of input should be (N, 3)."
        self._faces = faces

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        assert isinstance(index, list) or isinstance(index, np.ndarray), "Input should be list or numpy array"
        self._index = np.array(index)

    def get(self, key):
        """
        Get property of Geometry by key.

        Parameters
        ----------
        key: name of properties of Geometry, should be one of ['coords', 'faces', index']

        Return
        ------
        Value of properties.
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise ValueError('{} is not found.'.format(key))

    def set(self, key, value):
        """
        Set value to the property of Geometry by key.

        Parameters
        ----------
        key: name of properties of Geometry, should be one of ['coords', 'faces', index']
        value: value of properties that what to set.
        """
        if hasattr(self, key):
            self.__setattr__(key, value)
        else:
            raise ValueError('{} is not found.'.format(key))

    @property
    def centroid(self):
        """
        Get centroid of self geometry.

        Return
        ------
        cen: an instance of geometry class, centroid of self geometry.
        """
        # FIXME specify meaning of parameters.
        cen = Geometry(name=self.name)
        cen.coords = np.mean(self.coords, axis=0)
        cen.faces = None
        cen.index = None
        return cen

    def adjmatrix(self, sym=True):
        """
        Get adjacency matrix of Geometry, calculated from faces.

        Parameters
        ----------
        sym: make adjmatrix symmetrical, default is True.

        Returns
        -------
        adjm: adjacency matrix of (subj_id, hemi, surf), if mask=None, then shape = (n_vertexes, n_vertexes).
        """
        adjm = faces_to_adjmatrix(self.faces, sym=sym)
        return adjm

    def get_edges(self):
        """
        Get edges of Geometry, calculated from faces.

        Returns
        -------
        edges: array, edges of brain surface mesh, shape=(n_edges, 2).
        """
        edges = faces_to_edges(self.faces)
        return edges


