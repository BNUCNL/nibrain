#!/usr/bin/env python
#  tract class

from geometry import TractGeometry
from scalar import Scalar
from connection import Connection
from image import Image

class Tract(object):
    """
    Tract class that stores fiber data and provides analysis methods.

    Attributes:
    -----------
    name: name of tract, type: string
    source: source of region, type :string tck file path

    img: image attributes, such as space, itype, ras2vox, voxsize, dims

    geometry: geometry attributes of tract,such as xform, coordinates, lengths_mix, length_max,
              per_streamlines_id...
              It should be a instance of class Geometry_tract.

    scalar: scalar attributes, such as FA, MD...
            name: A list of attributes to identify scalar_value.
            scalar_value: MxN metric, M points of streamlines and value of N attributes of fiber tract.
            It should be an instance of class Scalar.

    connection:
    """

    def __init__(self,source,name=None):
        """
        Init Tract for further usage

        Parameter:
        ----------------------
        source: Tck file path, type: string
        name: name of tract, type: string
        """
        if name:
            assert isinstance(name,str), "Input name must be string."
        self.name = name
        if source:
            assert isinstance(source, str), "Input 'source' should be string."
        self._source = source

    @property
    def source(self):
        """Get the source of fiber tract."""
        return self._source

    @source.setter
    def source(self,set_source):
        """Set the source of fiber tract.
           Only first setting be permitted.
        """
        if not self._source:
            assert isinstance(set_source, str), "Input 'source' should be string."
            self._source = set_source
        else:
            raise ValueError("Source shouldn't be modified")

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry_load(self,source):
        source = self._source
        self._geometry = TractGeometry(source)

    @property
    def scalar(self):
        return self._scalar

    @scalar.setter
    def scalar(self):
        self._scalar = Scalar()

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self):
        self._connection = Connection()

    # @property
    # def img(self):
    #     return self._img
    #
    # @img.setter
    # def img(self):
    #     self._img = Image()