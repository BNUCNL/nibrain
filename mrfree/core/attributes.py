#!/usr/bin/env python

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
        """
        Get property of Geometry by key.

        Parameters
        ----------
            key: name of properties of GeometryAttribute, should be one of ['vertex_coords', 'vertex_faces', vertex_id']

        Return
        ------
            Value of properties.
        """
        if key == "name":
            return self.name
        elif key == "vertex_coords":
            return self.vertex_coords
        elif key == "vertex_faces":
            return self.vertex_faces
        elif key == "vertex_id":
            return self.vertex_id
        else:
            raise ValueError("Input should be one of ['name', 'vertex_coords', 'vertex_faces', 'vertex_id'].")

    def set(self, key, value):
        """
        Set value to the property of Geometry by key.

        Parameters
        ----------
            key: name of properties of GeometryAttribute, should be one of ['vertex_coords', 'vertex_faces', vertex_id']
            value: value of properties that what to set.
        """
        if key == "name":
            self.name = value
        elif key == "vertex_coords":
            self.vertex_coords = value
        elif key == "vertex_faces":
            self.vertex_faces = value
        elif key == "vertex_id":
            self.vertex_id = value
        else:
            raise ValueError("Input should be one of ['name', 'vertex_coords', 'vertex_faces', 'vertex_id'].")


class ScalarAttribute(object):
    """
    Class of ScalarAttribute.

    Attributes:
        name: A string or list as identity of scalar data.
        data: Scalar data.
    """

    def __init__(self, name = None, data = None):
        """
        Initialize the object.
        """
        if (name is not None) & (data is not None):
            self.set(name, data)
        else:
            print('Lack of input data.')

    def __setattr__(self, item, value):
        object.__setattr__(self, item, value)

    def __getattr__(self, item):
        if item not in self.__dict__:
            return None

    def __getitem__(self, item):
        return self.get(item)

    def __add__(self, other):
        sa_ins = ScalarAttribute()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) + other.get(idname))        
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __sub__(self, other):
        sa_ins = ScalarAttribute()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) - other.get(idname))
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __mul__(self, other):
        sa_ins = ScalarAttribute()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) * other.get(idname))
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __div__(self, other):
        sa_ins = ScalarAttribute()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) / other.get(idname))
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins
    
    def __abs__(self):
        self.data = np.abs(self.data)

    def __neg__(self):
        self.data = -1.0*self.data

    def __pos__(self):
        self.data = np.abs(self.data)

    def set(self, name, data):
        """
        Method to set identity of data and scalar data.

        Args:
            name: Identity of data.
            data: Scalar data.
        """
        if isinstance(name, str):
            name = [name]
        assert isinstance(name, list), "Name should be a string or list."
        assert isinstance(data, np.ndarray), "Convert data into np.ndarray before using it."
        if (len(np.unique(name)) == 1)&(len(name)<data.shape[1]):
            name = [name[0]]*data.shape[1]
        else:
            assert len(name)==data.shape[1], "Mismatch between name and data."    
        if data.ndim == 1:
            data = data[...,np.newaxis]
        if (self.name is None) | (self.data is None):
            self.name = name
            self.data = data
        else: 
            self.append(name, data) 

    def get(self, name):
        """
        Method to get scalar data by identity of data.

        Args:
            name: Identity of data.

        Returns:
            A scalar data (MxN, M vertices and N attributes) 
            that paired to name.
            
        Raises:
            pass    
        """
        if self.name is not None:
            # find all occurrences of name in a list of self.name
            indices = [i for i, x in enumerate(self.name) if x == name]
            if len(indices) == 0:
                print('Name mismatched.')
                return None
            else:
                return self.data[:, tuple(indices)]
        else:
            print('Set data firstly.')
            return None            

    def append(self, name = None, data = None):
        """
        A method to add scalar data in.

        Args:
            name: Identity of data.
            data: Scalar data.
        """
        if (name is not None) & (data is not None):
            assert isinstance(data, np.ndarray), "Convert data into np.ndarray before using it."
            if data.ndim == 1:
                data = data[...,np.newaxis]
            assert data.shape[0] == self.data.shape[0], "Array length mismatched."
            assert (self.name is not None) & (self.data is not None), "Please set name and data before appending it."
            if isinstance(name, str):
                name = [name]
            assert len(name) == data.shape[1], "Mismatch between name and data."
            self.name = self.name + name
            self.data = np.c_[self.data, data]
        else:
            print("Lack of input data.")
         
    def remove(self, name):
        """
        A method to delete scalar data which stored in the object.

        Args:
            name: Identity of data.
        """
        assert name in self.name, "Name mismatched."
        indices = [i for i, x in enumerate(self.name) if x == name]
        self.name = [x for i, x in enumerate(self.name) if i not in indices]
        self.data = np.delete(self.data, indices, axis=1)
        

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

    def get(self, key):
        """
        Get property of Geometry by key.

        Parameters
        ----------
            key: name of properties of GeometryAttribute, should be one of ['vertex_coords', 'vertex_faces', vertex_id']

        Return
        ------
            Value of properties.
        """
        if key == "region":
            return self.region
        elif key == "tract":
            return self.tract
        else:
            raise ValueError("Input should be one of ['region', 'tract'].")

    def set(self, key, value):
        """
        Set value to the property of Geometry by key.

        Parameters
        ----------
            key: name of properties of GeometryAttribute, should be one of ['vertex_coords', 'vertex_faces', vertex_id']
            value: value of properties that what to set.
        """
        if key == "region":
            self.region = value
        elif key == "tract":
            self.tract = value
        else:
            raise ValueError("Input should be one of ['region', 'tract'].")

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

