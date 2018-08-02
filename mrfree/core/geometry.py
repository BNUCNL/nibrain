#!/usr/bin/env python

# attributes class

import numpy as np

class Geometry(object):
    def __init__(self, data=None, id=None, gtype='volume',src=None):
        """
        Init Geometry.

        Parameters
        ----------
        data: the geometry data, a squeeze of array.
        id: the id for each array.
        gtype: geometry type, a string: volume, surface and streamline.
        src: source of the geometry data, a string.
        """
        self._gtype = ['volume', 'surface', 'streamline']

        self.gtype = gtype
        self.data  = data
        self.id = id
        self.src = src

    @property
    def gtype(self):
        return self._gtype

    @gtype.setter
    def gtype(self, gtype):
        assert isinstance(gtype, str), "Input 'gtype' should be string."
        assert gtype in self._surface_type, "gtype should be in {0}".format(self._surface_type)
        self._gtype = gtype

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        assert coords.ndim == 2, "Input should be 2-dim."
        assert coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._coords = coords

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        assert id.ndim == 2, "Input should be 2-dim."
        assert id.shape[1] == 3, "The shape of input should be (N, 3)."
        self._id = id

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



