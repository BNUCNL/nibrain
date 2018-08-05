#!/usr/bin/env python

import numpy as np
from base import intersect2d,exclude2d

class Points(object):
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

    def merge(self, other):
        assert isinstance(other, Points), "other should be a Points object"
        self.data = np.vstack(self.data)
        self.data = np.unique(self.data,axis=0)
        return self

    def intersect(self,other):
        assert isinstance(other, Points), "other should be a Points object"
        self.data = intersect2d(self.data, other.data)
        return self

    def exclude(self, other):
        assert isinstance(other, Points), "other should be a Points object"
        self.data = exclude2d(self.data, other.data)
        return self

    def centralize(self):
        self.data = np.mean(self.data,axis=0)
        return self

    def read_from_cifti(self):
        pass

    def save_to_cifti(self):
        pass

    def read_from_nifti(self):
        pass

    def save_to_nifti(self):
        pass

    def read_from_gifti(self):
        pass

    def save_to_gifti(self):
        pass


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
        assert isinstance(other, Lines), "other should be a Lines object"
        pass

    def equidistant_resample(self, num_segment):
        pass

    def skeleton(self):
        pass

    def read_from_tck(self):
        pass

    def save_to_tck(self):
        pass

    def read_from_trk(self):
        pass

    def save_to_trk(self):
        pass

    def read_from_vtk(self):
        pass

    def save_to_vtk(self):
        pass


class surface(object):
    def __init__(self):
        pass

if __name__ == "__main__":
    # Test Points
    data = np.random.rand(10,3)
    id = 1
    src = "Faked points"
    rg = Points(data, id, src)
    rg.data = np.random.rand(5,3)
    rg.id = 2
    rg.src = "New faked points"

    # Test Lines
    data = [np.array([[0, 0., 0.9],
                      [1.9, 0., 0.]]),
            np.array([[0.1, 0., 0],
                      [0, 1., 1.],
                      [0, 2., 2.]]),
            np.array([[2, 2, 2],
                      [3, 3, 3]])]
    id = np.arange(len(data))
    src = "Faked lines"
    lg = Lines(data, id, src)
    lg.data =  rg.data.remove(1)
    lg.id = np.delete(rg.id,1)
    lg.src = "New faked lines"


