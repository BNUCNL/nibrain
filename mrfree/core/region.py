#!/usr/bin/env python

# region class

from geometry import Points
from scalar import Scalar
from image import Image
from connection import Connection

class Region(object):
    """ A class for brain region/area analysis.

    Attributes 
    ----------
    name: a str, region name
    ia: Image object, image attributes of the region
    ga: Points object, geometry attributs of the region
    sa: Scalar object, scalar attributes of the region.
    ca: Connection object, connection attributes of the region
    """
    def __init__(self, name, ia=None, ga=None, sa=None, ca=None):
        """ init the region with image, geometry, scalar and connection attributes
        
        Parameters 
        ----------
        name: a str, region name
        ia: Image object, image attributes of the region
        ga: Points object, geometry attributs of the region
        sa: Scalar object, scalar attributes of the region.
        ca: Connection object, connection attributes of the region
        """
        
        self.name = name
        self.ia = ia
        self.ga = ga
        self.sa = sa
        self.ca = ca

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), "Input 'name' should be string."
        self._name = name

    @property
    def ia(self):
        return self._ia

    @ia.setter
    def ia(self, ia):
        assert isinstance(ia, Image), "ia should be a Image object"
        self._ia = ia

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, ga):
        assert isinstance(ga, Points), "ga should be a RegionGeometry obejct."
        self._ga = ga

    @property
    def sa(self):
        return self._sa

    @sa.setter
    def sa(self, sa):
        assert isinstance(sa, Scalar), "sa should be a Scalar object"
        self._sa = sa

    @property
    def ca(self):
        return self._ca

    @ca.setter
    def ca(self, ca):
        assert isinstance(ca, Connection), "ca should be a Connection object."
        self._ca = ca

    def merge(self, other):
        """ Merge other region into the region.

        Parameters
        ----------
        other: a Region object, another region

        Return
        ----------
        self: merged region
        """
        assert isinstance(other, Region), "other should be a other obejct."
        if hasattr(self, 'ga'):
            self.ga = self.ga.merge(other.ga)
            if hasattr(self, 'sa'):
                self.sa = self.sa.append(None, other.sa)

        return  self

    def intersect(self, other):
        """ Intersect with other region

        Parameters
        ----------
        other: a Region object, another region

        Return
        ----------
        self: the intersected region
        """

        assert isinstance(other, Region), "other should be a other obejct."
        if hasattr(self, 'ga'):
            self.ga, idx = self.ga.intersect(other.ga)
            if hasattr(self, 'sa'):
                self.sa = self.sa.remove(idx)

        return self

    def exclude(self, other):
        """ Exclude other region from the region.

        Parameters
        ----------
        other: a Region object, another region

        Return
        ----------
        self: the left region
        """

        assert isinstance(other, Region), "other should be a other obejct."
        if hasattr(self, 'ga'):
            self.ga, idx = self.ga.exclude(other)
            if hasattr(self, 'sa'):
                self.sa = self.sa.remove(idx)

            if hasattr(self, 'connection'):
                pass

        return  self

    @property
    def centralize(self):
        if hasattr(self, 'ga'):
            self.ga = self.ga.centralize()
        if hasattr(self,'sa'):
            self.sa = self.sa.mean()

        return self