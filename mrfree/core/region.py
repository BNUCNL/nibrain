# region class
from mrfree.core.attributes import Geometry, Scalar, Connection


class Region(object):
    """
    Region class that stores data and provides analysis methods.

    Parameters
    ----------
        name: name of region, type: string.
        layer: layer number of region, type: string.
        source: source of region, type: string.
        space: space of region, type: string

        xform: transform matrix of region
        anat_coords: coords of region, should be N*3 array.
        geometry: geometry attributes, should be an instance of class Geometry.
        scalar: scalar attributes, should be an instance of class Scalar.
        connection: connection attributes, should be an instance of class Connection.
    """
    def __init__(self, name, layer='4', source=None, space='native'):
        """
        Init Region for further usage.

        Parameters
        ----------
            name: name of region, type: string.
            layer: layer number of region, type: string.
            source: source of region, type: string.
            space: space of where this region exists, type: string
        """
        self.name = name
        self.layer = layer
        self.source = source
        self.space = space

    @property
    def name(self):
        """
        Get name of region.

        Return
        ------
            Name of region.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Set name of region, input should be string.

        Parameters
        ----------
            name: name of region, type: string.
        """
        assert isinstance(name, str), "Input 'name' should be string."
        self._name = name

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = str(layer)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        assert isinstance(source, str), "Input 'source' should be string."
        self._source = source

    @property
    def xform(self):
        return self._xform

    @xform.setter
    def xform(self, xform):
        assert xform.shape == (4, 4), "Shape of xform should be (4, 4)"
        self._xform = xform

    @property
    def anat_coords(self):
        return self._anat_coords

    @anat_coords.setter
    def anat_coords(self, anat_coords):
        assert anat_coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._anat_coords = anat_coords

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        assert isinstance(geometry, Geometry), "Input 'geometry' should be an instance of Geometry."
        self._geometry = geometry

    @property
    def scalar(self):
        return self._scalar

    @scalar.setter
    def scalar(self, scalar):
        assert isinstance(scalar, Scalar), "Input 'scalar' should be an instance of Scalar."
        self._scalar = scalar

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, connection):
        assert isinstance(connection, Connection), "Input 'connection' should be an instance of Connection."
        self._connection = connection

