#!/usr/bin/env python

from geometry import Lines
from scalar import Scalar


class Tract(object):
    """ Tract class was designed to represent data related to white matter tract of the brain.

    Attributes 
    ----------
    ga: Lines object, geometry attributs of the tract
    sa: Scalar object, scalar attributes of the tract.
    """

    def __init__(self, ga=None, sa=None):
        """ init the tract with image, geometry, and scalar attributes

        Parameters 
        ----------
        ga: Lines object, geometry attributs of the tract
        sa: Scalar object, scalar attributes of the tract.
        """
        self.ga = ga
        self.sa = sa

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

    def create_from_scratch(self, gs=None, ss=None, toi=None):
        """ Create tract object from geometry and scalar source data

        Parameters
        ----------
        gs: Tractogram object, source for geometry attributes
        ss: Image or Tractogram object, source for scalar attributes

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