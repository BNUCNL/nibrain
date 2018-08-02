#!/usr/bin/env python

# attributes class
import numpy as np
import dipy.tracking.streamline.ArraySequence as ArraySequence

class RegionGeometry(object):
    def __init__(self, data=None, id=None, src=None):
        """
        Init Geometry.

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

    def merge(self, another_geo):
        assert isinstance(another_geo, RegionGeometry), "another_geo should be the same class"
        pass

    def intersect(self,another_geo):
        assert isinstance(another_geo, RegionGeometry), "another_geo should be the same class"
        pass

    def subtract(self):
        pass

    def center(self):
        pass


class TractGeometry(object):
    def __init__(self, data=None, id=None, src=None):
        """
        Init Geometry.

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


def merge(self, another_geo):
        assert isinstance(another_geo, TractGeometry),"another_geo should be the same class"
        pass


    def equidistant_resample(self):
        pass


    def skeleton(self):
        pass


if __name__ == "__main__":
    pass



