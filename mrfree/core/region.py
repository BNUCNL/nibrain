# region class
import numpy as np
from mrfree.core.attributes import Geometry, Scalar, Connection
from mrfree.io import load


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
        if source:
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

    def __update(self, index):
        """Update data by anat_coords."""
        if hasattr(self, 'geometry'):
            self.geometry.coords = self.geometry.coords[index]
            self.geometry.faces = self.geometry.faces[index]
            self.geometry.index = self.geometry.index[index]

        if hasattr(self, 'scalar'):
            self.scalar.data = self.scalar.data[index]

        if hasattr(self, 'connection'):
            pass

    def union(self, region):
        """
        Merge another region into self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."

        # FIXME define unique_index
        unique_coords = [item for item in region.anat_coords if item not in self.anat_coords]
        unique_index =
        self.anat_coords, index = self.__union_anat_coords(region.anat_coords)

        if hasattr(self, 'geometry'):
            self.geometry.coords = self.geometry.coords[index] + region.geometry.coords[unique_index]
            self.geometry.faces = self.geometry.faces[index] + region.geometry.faces[unique_index]
            self.geometry.index = self.geometry.index[index] + region.geometry.index[unique_index]

        if hasattr(self, 'scalar'):
            self.scalar.data = self.scalar.data[index].append(region.scalar.data[uindex])

        if hasattr(self, 'connection'):
            pass

    def __union_anat_coords(self, anat_coords):
        anat_coords = np.append(self.anat_coords, anat_coords, axis=0)
        return np.unique(anat_coords, axis=0), index

    def intersect(self, region):
        """
        Intersect another region into self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."

        self.__anat_coords, index = self.__intersect_anat_coords(region.anat_coords)
        self.__update(index)

    def __intersect_anat_coords(self, anat_coords):
        result = []
        for i in self.anat_coords:
            for j in anat_coords:
                if np.all(i == j):
                    result.append(i)
        return np.array(result), index

    def exclude(self, region):
        """
        Exclude another region out of self.

        Parameters
        ----------
            region: an instance of Region class, its layer and space should be the same as this region class.
        """
        assert self.layer == region.layer, "Layer of regions do not match."
        assert self.space == region.space, "Space of regions do not match."

        self.__anat_coords, index = self.__exclude_anat_coords(region.anat_coords)
        self.__update(index)

    def __exclude_anat_coords(self, anat_coords):
        result = []
        for i in self.anat_coords:
            match = 0
            for j in anat_coords:
                if np.all(i == j):
                    match = 1
                    break
            if not match:
                result.append(i)
        return np.array(result), index

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
