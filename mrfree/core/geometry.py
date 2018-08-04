#!/usr/bin/env python

import numpy as np

class RegionGeometry(object):
    def __init__(self, data=None, id=None, src=None):
        """
        Parameters
        ----------
        data: geometry data, a sequence of array.
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
        self.data = np.vstack(self.data)
        self.data = np.unique(self.data,axis=0)
        return self

    def intersect(self,rg):
        assert isinstance(rg, RegionGeometry), "rg should be a RegionGeometry object"
        self.data = intersect2d(self.data, rg.data)
        return self

    def exclude(self):
        assert isinstance(rg, RegionGeometry), "rg should be a RegionGeometry object"
        self.data = exclude2d(self.data, rg.data)
        return self


    def centralize(self):
        self.data = np.mean(self.data,axis=0)
        return self


class TractGeometry(object):
    def __init__(self,source=None):
        """
        Parameters
        ----------
        data: geometry data, a sequence of array.
        id: the id for each array.
        source: source of the geometry data, a string.
        """
        self._src = source
        self._data = None
        self._id = None
        self._shape = None

    @property
    def source(self):
        return self._src

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

def merge(self, tg):
    assert isinstance(tg, TractGeometry),"tg should be a TractGeometry object"
    pass

def equidistant_resample(self):
    pass

def skeleton(self):
    pass


if __name__ == "__main__":
    # Test RegionGeometry
    data = np.random.rand(10,3)
    id = 1
    src = "Faked region geometry"
    rg = RegionGeometry(data, id,src)
    rg.data = np.random.rand(5,3)
    rg.id = 2
    rg.src = "New faked region geometry"

    # Test TractGeometry
    data = [np.array([[0, 0., 0.9],
                      [1.9, 0., 0.]]),
            np.array([[0.1, 0., 0],
                      [0, 1., 1.],
                      [0, 2., 2.]]),
            np.array([[2, 2, 2],
                      [3, 3, 3]])]
    id = np.arange(len(data))
    src = "Faked tract geometry"
    rg = TractGeometry(data, id, src)
    rg.data =  rg.data.remove(1)
    rg.id = np.delete(rg.id,1)
    rg.src = "New faked tract geometry"


