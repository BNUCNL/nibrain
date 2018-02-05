# region class
import re
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
        self.name = name
        self.layer = layer
        self.source = source
        self.space = space

        self.xform = None
        self.anat_coords = None
        self.ga = None
        self.sa = None
        self.ca = None

    def get_name(self):
        """
        Get name of region.

        Return
        ------
            Name of region.
        """
        return self.name

    def set_name(self, name):
        """
        Set name of region, input should be string.

        Parameters
        ----------
            name: name of region, type: string.
        """
        if not isinstance(name, str):
            raise ValueError("Input 'name' should be string.")
        self.name = name

    def get_layer(self):
        return self.layer

    def set_layer(self, layer):
        layer = str(layer)
        if not re.findall('[1-6]', layer):
            raise ValueError("Input 'layer' should contain at least one number.")
        self.layer = layer

    def get_source(self):
        return self.source

    def set_source(self, source):
        if not isinstance(source, str):
            raise ValueError("Input 'source' should be string.")
        self.source = source

    def get_xform(self):
        return self.xform

    def set_xform(self, xform):
        self.xform = xform

    def get_anat_coords(self):
        return self.anat_coords

    def set_anat_coords(self, anat_coords):
        if anat_coords.shape[1] != 3:
            raise ValueError("The shape of input should be [N, 3].")
        self.anat_coords = anat_coords

    def get_ga(self):
        return self.ga

    def set_ga(self, ga):
        if not isinstance(ga, GeometryAttribute):
            raise ValueError("Input 'ga' should be an instance of GeometryAttribute.")
        self.ga = ga

    def get_sa(self):
        return self.sa

    def set_sa(self, sa):
        if not isinstance(ga, ScalarAttribute):
            raise ValueError("Input 'sa' should be an instance of ScalarAttribute.")
        self.sa = sa

    def get_ca(self):
        return self.ca

    def set_ca(self, ca):
        if not isinstance(ga, ConnectionAttribute):
            raise ValueError("Input 'ca' should be an instance of ConnectionAttribute.")
        self.ca = ca

