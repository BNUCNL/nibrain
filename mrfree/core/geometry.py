#!/usr/bin/env python

# attributes class

import numpy as np

class Geometry(object):
    def __init__(self, data=None, id=None, gtype='volume', src=None):
        """
        Init Geometry.

        Parameters
        ----------
        data: geometry data, a squeeze of array.
        id: the id for each array.
        gtype: geometry type, a string: volume and streamline.
        src: source of the geometry data, a string.
        """
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
        known_gtype = ('volume', 'streamline')
        assert gtype in known_gtype, "gtype should be in {0}".format(known_gtype)
        self._gtype = gtype

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert data.ndim == 2, "Input should be 2-dim."
        assert data.shape[1] == 3, "The shape of input should be (N, 3)."
        self._data = data

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        assert id.ndim == 2, "Input should be 2-dim."
        assert id.shape[1] == 3, "The shape of input should be (N, 3)."
        self._id = id

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        assert isinstance(src, list) or isinstance(src, np.ndarray), "Input should be list or numpy array"
        self._src = np.array(src)



    def skeletonize(self):
        """
        Get centroid of self geometry.

        Return
        ------
        cen: an instance of geometry class, centroid of self geometry.
        """
        pass

    def merge(self):
        pass

    def intersect(self):
        pass

    def subtract(self):
        pass

    def equidistant_resample(self):
        pass







