# attributes class
import numpy as np


class GeometryAttribute(object):
    def __init__(self, name, vertex_coords=None, vertex_faces=None, vertex_id=None):
        """
        Init GeometryAttribute.

        Parameters
        ----------
            name: the name of where geometry indicated, like 'inflated', 'sphere' etc.
            vertex_coords: coords of vertexes, should be N*3 array.
            vertex_faces: faces of vertexes, should be M*3 array.
            vertex_id: vertexes id in this geometry, should be K*1 array.
        """
        self._surface_type = ['white', 'pial', 'inflated', 'sphere']

        self.name = name
        self.vertex_coords = vertex_coords
        self.vertex_faces = vertex_faces
        self.vertex_id = vertex_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), "Input 'name' should be string."
        assert name in self._surface_type, "Name should be in {0}".format(self._surface_type)
        self._name = name

    @property
    def vertex_coords(self):
        return self._vertex_coords

    @vertex_coords.setter
    def vertex_coords(self, vertex_coords):
        assert vertex_coords.ndim == 2, "Input should be 2-dim."
        assert vertex_coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._vertex_coords = vertex_coords

    @property
    def vertex_faces(self):
        return self._vertex_faces

    @vertex_faces.setter
    def vertex_faces(self, vertex_faces):
        assert vertex_faces.ndim == 2, "Input should be 2-dim."
        assert vertex_faces.shape[1] == 3, "The shape of input should be (N, 3)."
        self._vertex_faces = vertex_faces

    @property
    def vertex_id(self):
        return self._vertex_id

    @vertex_id.setter
    def vertex_id(self, vertex_id):
        assert isinstance(vertex_id, list) or isinstance(vertex_id, np.ndarray), "Input should be list or numpy array"
        self._vertex_id = vertex_id

    def get(self, key):
        if key == "name":
            return self._name
        elif key == "vertex_coords":
            return self._vertex_coords
        elif key == "vertex_faces":
            return self._vertex_faces
        elif key == "vertex_id":
            return self._vertex_id
        else:
            raise ValueError("Input should be in ['vertex_coords', 'vertex_faces', vertex_id'].")

    def set(self, key, value):
        if key == "name":
            assert isinstance(value, str), "Input 'name' should be string."
            assert value in self._surface_type, "Name should be in {0}".format(self._surface_type)
            self._name = value

        elif key == "vertex_coords":
            assert value.ndim == 2, "Input should be 2-dim."
            assert value.shape[1] == 3, "The shape of input should be (N, 3)."
            self._vertex_coords = value

        elif key == "vertex_faces":
            assert value.ndim == 2, "Input should be 2-dim."
            assert value.shape[1] == 3, "The shape of input should be (N, 3)."
            self._vertex_faces = value

        elif key == "vertex_id":
            assert isinstance(value, list) or isinstance(value, np.ndarray), "Input should be list or numpy array"
            self._vertex_id = value
        else:
            raise ValueError("Input should be one of ['name', 'vertex_coords', 'vertex_faces', vertex_id'].")


class ConnectionAttribute(object):
    def __init__(self, region=None, tract=None):
        """
        Init ConnectionAttribute.

        Parameters
        ----------
            region: a list of regions.
            tract: a list of tracts
        """
        self.region = region
        self.tract = tract

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        self._region = region

    @property
    def tract(self):
        return self._tract

    @tract.setter
    def tract(self, tract):
        self._tract = tract

    def append(self, ca):
        """
        Merge another ConnectionAttribute class.

        Parameters
        ----------
            ca: an instance of ConnectionAttribute class.
        """
        self.region.append(ca.region)
        self.tract.append(ca.tract)

    def remove(self):
        pass

