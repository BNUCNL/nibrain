#!/usr/bin/env python

# region class

from geometry import Points
from scalar import Scalar
from connection import Connection
from mrfree.mrfree.io import load

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
    def __init__(self, name, ia=None, ga=None, sa=None, ca=None):
        """
        Init Region for further usage.

        Parameters
        ----------
        name: name of region, type: string.
        """
        self.name = name
        self.ia = ia
        self.ga = ga
        self.sa = sa
        self.ca = ca

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

    def merge(self, region):
        """
        Merge another region into self.

        Parameters
        ----------
        region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert isinstance(region, Region), "region should be a Region obejct."
        if hasattr(self, 'ga'):
            self.ga = self.ga.merge(region.ga)
            if hasattr(self, 'sa'):
                self.sa = self.sa.append(None, region.sa)

        return  self

    def intersect(self, region):
        assert isinstance(region, Region), "region should be a Region obejct."
        if hasattr(self, 'ga'):
            self.ga, idx = self.ga.intersect(region.ga)
            if hasattr(self, 'sa'):
                self.sa = self.sa.remove(idx)

        return self

    def exclude(self, region):
        assert isinstance(region, Region), "region should be a Region obejct."
        if hasattr(self, 'ga'):
            self.ga, idx = self.ga.exclude(region)
            if hasattr(self, 'sa'):
                self.sa = self.sa.remove('xx',idx)

        if hasattr(self, 'connection'):
            pass

        return  self

    @property
    def centralize(self):
        if hasattr(self, 'ga'):
            self.ga = self.ga.centralize()
        if hasattr(self,'sa'):
            self.sa = self.sa.xx

        return self



class SurfaceRegion(Region):
    """

    """
    def load_geometry(self, name, surf_file, surf_label_file=None):
        """
        Load surf info into Geometry by load function.

        Parameters
        ----------
        name: the name of where geometry indicated, like 'inflated', 'sphere' etc.
        surf_file: Surface file path, specified as a filename (single file).
        surf_label_file: Surface label file path, specified as a filename (single file).
        """
        coords, faces, label = load.load_surf_geom(surf_file, surf_label_file)
        self.geometry = Geometry(name, coords, faces, label)

    def load_scalar(self, name, surf_file, surf_label_file=None):
        """
        Load scalar data into Scalar by load function.

        Parameters
        ----------
        name: A string or list as identity of scalar data.
        surf_file: Surface file path, specified as a filename (single file).
        surf_label_file: Surface label file path, specified as a filename (single file).
        """
        data = load.load_surf_scalar(surf_file, surf_label_file)
        self.scalar = Scalar(name, data)

    def save(self, save_path):
        pass


class VolumeRegion(Region):
    """

    """
    def load_geometry(self, vol_file, vol_mask_file=None):
        """
        Load volume geometry by load function.

        Parameters
        ----------
        vol_file : Volume file path. Nifti dataset, specified as a filename (single file).
        vol_mask_file: Volume mask file path. Nifti dataset, specified as a filename (single file).
        """
        coords, xform = load.load_vol_geom(vol_file, vol_mask_file)
        self.xform = xform
        self.anat_coords = coords

    def load_scalar(self, name, vol_file, vol_mask_file=None):
        """
        Load volume scalar by load function.

        Parameters
        ----------
        name: A string or list as identity of scalar data.
        vol_file : Volume file path. Nifti dataset, specified as a filename (single file).
        vol_mask_file: Volume mask file path. Nifti dataset, specified as a filename (single file).
        """
        data = load.load_vol_scalar(vol_file, vol_mask_file)
        self.scalar = Scalar(name, data)

    def save(self, save_path):
        pass
