#!/usr/bin/env python

import numpy as np
import dipy.tracking.streamline.ArraySequence as ArraySequence


class RegionGeometry(object):
    def __init__(self, data=None, id=None, src=None):
        """
        Parameters
        ----------
        data: geometry data, a squeeze of array.
        id: the id for each array.
        src: source of the geometry data, a string.
        """
        self.data  = data
        self.id = id
        self.src = src

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

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        self._src = src

    def merge(self, rg):
        assert isinstance(rg, RegionGeometry), "rg should be a RegionGeometry object"
        pass

    def intersect(self,rg):
        assert isinstance(rg, RegionGeometry), "rg should be a RegionGeometry object"
        pass

    def subtract(self):
        pass

    def center(self):
        pass


class TractGeometry(object):
    def __init__(self, data=None, id=None, src=None):
        """
        Parameters
        ----------
        data: geometry data, a squeeze of array.
        id: the id for each array.
        src: source of the geometry data, a string.
        """
        self.data = data
        self.id = id
        self.src = src

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

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        self._src = src


def merge(self, tg):
    assert isinstance(tg, TractGeometry),"tg should be a TractGeometry object"
    pass

    def equidistant_resample(self):
        pass

    def skeleton(self):
        pass

if __name__ == "__main__":
    pass



