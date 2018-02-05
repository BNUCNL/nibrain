# region class
from .attributes import GeometryAttribute, ScalarAttribute, ConnectionAttribute


class Region(object):
    """
    Region class that stores data and provides analysis methods.

    Parameters
    ----------
        name: name of region, type: string.
        layer: layer number of region, type: string.
        source: source of region, type: string.
        space: space of region, type:string

        xform: transform matrix of region
        anat_coords: coords of region, should be N*3 array.
        ga: geometry attributes.
        sa: scalar attributes.
        ca: connection attributes.
    """
    def __init__(self, name, layer=None, source=None, space=None):
        """
        Init Region for further usage.

        Parameters
        ----------
            name: name of region, type: string.
            layer: layer number of region, type: string.
            source: source of region, type: string.
            space: space of region, type:string
        """
        self._name = name
        self._layer = layer
        self._source = source
        self._space = space

        self._xform = None
        self._anat_coords = None
        self._ga = None
        self._sa = None
        self._ca = None

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
        layer = str(layer)
        self._layer = layer

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
        self.xform = xform

    @property
    def anat_coords(self):
        return self._anat_coords

    @anat_coords.setter
    def anat_coords(self, anat_coords):
        assert anat_coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._anat_coords = anat_coords

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, ga):
        assert isinstance(ga, GeometryAttribute), "Input 'ga' should be an instance of GeometryAttribute."
        self._ga = ga

    @property
    def sa(self):
        return self._sa

    @sa.setter
    def sa(self, sa):
        assert isinstance(sa, ScalarAttribute), "Input 'sa' should be an instance of ScalarAttribute."
        self._sa = sa

    @property
    def ca(self):
        return self._ca

    @ca.setter
    def ca(self, ca):
        assert isinstance(ca, ConnectionAttribute), "Input 'ca' should be an instance of ConnectionAttribute."
        self._ca = ca

