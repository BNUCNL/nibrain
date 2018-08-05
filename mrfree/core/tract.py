#!/usr/bin/env python
#  tract class

from geometry import Lines
from scalar import Scalar
from connection import Connection
from image import Image


class Tract(object):
    """ Tract class was designed to represent data related to white matter tract of the brain.

    Attributes 
    ----------
    name: a str, tract name
    ia: Image object, image attributes of the tract
    ga: Lines object, geometry attributs of the tract
    sa: Scalar object, scalar attributes of the tract.
    """

    def __init__(self, name, ia=None, ga=None, sa=None):
        """ init the tract with image, geometry, scalar attributes

        Parameters 
        ----------
        name: a str, tract name
        ia: Image object, image attributes of the tract
        ga: Lines object, geometry attributs of the tract
        sa: Scalar object, scalar attributes of the tract.
        ca: Connection object, connection attributes of the tract
        """

        self.name = name
        self.ia = ia
        self.ga = ga
        self.sa = sa

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), " name should be string."
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
        assert isinstance(ga, Lines), "ga should be a Lines obejct."
        self._ga = ga

    @property
    def sa(self):
        return self._sa

    @sa.setter
    def sa(self, sa):
        assert isinstance(sa, Scalar), "sa should be a Scalar object"
        self._sa = sa

    def merge(self, other, axis=0):
        """ Merge other tract into the tract based on the line id from geometry attributes.

        Parameters
        ----------
        other: a tract object, another tract
        axis: integer, 0 or 1

        Return
        ----------
        self: merged tract
        """
        assert isinstance(other, Tract), "other should be a Tract obejct."
        if axis == 0:  # merge both ga and sa in rows
            if hasattr(self, 'ga'):
                self.ga = self.ga.merge(other.ga)
            if hasattr(self, 'sa'):
                self.sa = self.sa.append(other.sa)
        else:  # only merge sa in column
            if hasattr(self, 'sa'):
                self.sa = self.sa.join(other.sa)

        return self

    def intersect(self, other):
        """ Intersect with other tract based on the line id from geometry attributes.

        Parameters
        ----------
        other: a tract object, another tract

        Return
        ----------
        self: the intersected tract
        """

        assert isinstance(other, Tract), "other should be a Tract obejct."
        pass

    def exclude(self, other):
        """ Exclude other tract from the tract based on the line id from geometry attributes.

        Parameters
        ----------
        other: a tract object, another tract

        Return
        ----------
        self: the left tract
        """

        assert isinstance(other, Tract), "other should be a Tract obejct."
        pass

    def skeleton(self):
        if hasattr(self, 'ga'):
            self.ga = self.ga.skeleton()
        if hasattr(self, 'sa'):
            self.sa = self.sa.mean()

        return self

    def create_from_scratch(self, ref_image=None, scalar_image=None, tractograph=None):
        """ Create tract object from raw data which contain the image, geometry and scalar information of the tract

        Parameters
        ----------
        ref_image: a nifit image pathstr or a Image object
            The refer image for the tract
        scalar_image: a nifti image pathstr or a Scalar object
            The scalar image, representing some scalar information of the tract
        tractograph: a tractograph pathstr or a Lines object
            The tractograph from fiber tracking, carrying the geometry information of the tract

        Returns
        -------
        self: created tract object
        """
        pass

    def load(self, filename):
        """ Load tract object from serializing persistence file(Jason or pickle file)

        Parameters
        ----------
        filename: str
            File pathstr to a tract serializing persistence file
        Returns
        -------
        self: Tract object

        """
        pass

    def save(self, filename):
        """ save tract object to a serializing persistence file(Jason or pickle file)

        Parameters
        ----------
        filename: str
            File pathstr to a tract serializing persistence file

        Returns
        -------

        """

        pass
