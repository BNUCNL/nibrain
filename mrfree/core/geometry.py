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

    """Test RegionGeometry"""
    # creat object
    data = np.random.rand(10,3)
    id = 1
    src = "Faked region geometry"
    rg = RegionGeometry(data, id,src)

    # test set and get
    rg.data = np.random.rand(5,3)
    rg.id = 2
    rg.src = "New faked region geometry"

    """Test TractGeometry"""
    # creat object
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

    # test set and get
    rg.data =  rg.data.remove(1)
    rg.id = np.delete(rg.id,1)
    rg.src = "New faked tract geometry"



