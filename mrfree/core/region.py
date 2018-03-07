# region class
import numpy as np
from mrfree.core.attributes import Geometry, Scalar, Connection
from mrfree.io import io


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
    def __init__(self, name, layer='L1-6', source=None, space='native'):
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

    def union(self, region):
        """
        Merge another region into self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."

        self.__union_anat_coords(region.anat_coords)
        if hasattr(self, 'geometry'):
            self.__union_geometry(region.geometry)
        self.__union_scalar(region.scalar)
        self.__union_connection(region.connection)

    def __union_anat_coords(self, anat_coords):
        anat_coords = np.append(self.anat_coords, anat_coords, axis=0)
        self.anat_coords = np.unique(anat_coords, axis=0)

    def __union_geometry(self, geometry):
        self.geometry.union(geometry)

    def __union_scalar(self, scalar):
        # FIXME union scalar may need to remove duplicate elements.
        self.scalar.aggregate(scalar)

    def __union_connection(self, connection):
        # FIXME union connection may need to remove duplicate elements.
        self.connection.append(connection)

    def intersect(self, region):
        """
        Intersect another region into self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."

        self.__intersect_anat_coords(region.anat_coords)
        if hasattr(self, 'geometry'):
            self.__intersect_geometry(region.geometry)
        self.__intersect_scalar(region.scalar)
        self.__intersect_connection(region.connection)

    def __intersect_anat_coords(self, anat_coords):
        result = []
        for i in self.anat_coords:
            for j in anat_coords:
                if np.all(i == j):
                    result.append(i)
        self.anat_coords = np.array(result)

    def __intersect_geometry(self, geometry):
        self.geometry.intersect(geometry)

    def __intersect_scalar(self, scalar):
        pass

    def __intersect_connection(self, connection):
        pass

    def exclude(self, region):
        """
        Exclude another region out of self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        # TODO to be completed.
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."
        self.exclude_anat_coords(region.anat_coords)
        if hasattr(self, 'geometry'):
            self.exclude_geometry(region.geometry)

        self.exclude_scalar(region.scalar)
        self.exclude_connection(region.connection)

    def exclude_anat_coords(self, anat_coords):
        result = []
        for i in self.anat_coords:
            match = 0
            for j in anat_coords:
                if np.all(i == j):
                    match = 1
                    break
            if not match:
                result.append(i)
        self.anat_coords = np.array(result)

    def exclude_geometry(self, geometry):
        self.geometry.exclude(geometry)

    def exclude_scalar(self, scalar):
        # FIXME the part of getting data should be modified.
        data1, name1 = self.scalar.get('all')
        __, name2 = scalar.get('all')

        result_name = list(np.array(name1)[np.in1d(name1, name2)])
        result_data = data1[np.in1d(name1, name2)]

        self.scalar.name = result_name
        self.scalar.data = result_data

    def exclude_connection(self, connection):
        pass

    @property
    def centroid(self):
        """
        Calculate centroid of region in its property.

        Return
        ------
            cen: region class that contain properties of center point.
        """
        cen = Region(name=self.name, layer=self.layer, source=self.source, space=self.space)
        cen.xform = self.xform
        cen.anat_coords = self.centroid_anat_coords
        if hasattr(self, 'geometry'):
            cen.geometry = self.geometry.centroid

        # TODO scalar.centroid, connection.centroid
        cen.scalar = self.scalar.centroid
        cen.connection = self.connection.centroid
        return cen

    @property
    def centroid_anat_coords(self):
        """
        Calculate centroid of region's anat_coords.

        Return
        ------
            cen_anat_coords: centroid of anat_coords in region.
        """
        # TODO specify mean method.
        cen_anat_coords = np.mean(self.anat_coords, axis=1)
        return cen_anat_coords


class SurfaceRegion(Region):
    """

    """
    def load_geometry(self, surf_file):
        # TODO specify index and label
        coords, faces, label = io.load_surf_geom(surf_file)
        self.geometry.coords = coords
        self.geometry.faces = faces
        self.geometry.index = label

    def load_scalar(self, name, surf_file, label=None):
        data = io.load_surf_scalar(surf_file, label)
        self.scalar.set(name, data)

    def save(self, save_path):
        pass


class VolumeRegion(Region):
    """

    """
    def load_geometry(self, vol_file):
        coords, xform = io.load_vol_geom(vol_file)
        self.xform = xform
        self.anat_coords = coords

    def load_scalar(self, name, vol_file, label=None):
        data = io.load_vol_scalar(vol_file, label)
        self.scalar.set(name, data)

    def save(self, save_path):
        pass
