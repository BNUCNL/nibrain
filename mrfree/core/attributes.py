#!/usr/bin/env python

# attributes class

import numpy as np


class Geometry(object):
    def __init__(self, name, coords=None, faces=None, index=None):
        """
        Init Geometry.

        Parameters
        ----------
            name: the name of where geometry indicated, like 'inflated', 'sphere' etc.
            coords: coords of vertexes, should be N*3 array.
            faces: faces of vertexes, should be M*3 array.
            index: vertexes index in this geometry, should be K*1 array.
        """
        self._surface_type = ['white', 'pial', 'inflated', 'sphere']

        self.name = name
        self.coords = coords
        self.faces = faces
        self.index = index

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str), "Input 'name' should be string."
        assert name in self._surface_type, "Name should be in {0}".format(self._surface_type)
        self._name = name

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        assert coords.ndim == 2, "Input should be 2-dim."
        assert coords.shape[1] == 3, "The shape of input should be (N, 3)."
        self._coords = coords

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        assert faces.ndim == 2, "Input should be 2-dim."
        assert faces.shape[1] == 3, "The shape of input should be (N, 3)."
        self._faces = faces

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        assert isinstance(index, list) or isinstance(index, np.ndarray), "Input should be list or numpy array"
        self._index = np.array(index)

    def get(self, key):
        """
        Get property of Geometry by key.

        Parameters
        ----------
            key: name of properties of Geometry, should be one of ['coords', 'faces', index']

        Return
        ------
            Value of properties.
        """
        if key == "name":
            return self.name
        elif key == "coords":
            return self.coords
        elif key == "faces":
            return self.faces
        elif key == "index":
            return self.index
        else:
            raise ValueError("Input should be one of ['name', 'coords', 'faces', 'index'].")

    def set(self, key, value):
        """
        Set value to the property of Geometry by key.

        Parameters
        ----------
            key: name of properties of Geometry, should be one of ['coords', 'faces', index']
            value: value of properties that what to set.
        """
        if key == "name":
            self.name = value
        elif key == "coords":
            self.coords = value
        elif key == "faces":
            self.faces = value
        elif key == "index":
            self.index = value
        else:
            raise ValueError("Input should be one of ['name', 'coords', 'faces', 'index'].")

    def union(self, geometry):
        """
        Merge another geometry into this.

        Parameters
        ----------
            geometry: an instance of Geometry class.
        """
        self.__union_faces(geometry.faces)
        self.__union_coords(geometry.coords)
        self.__union_index(geometry.index)

    def __union_faces(self, faces):
        faces = np.append(self.faces, faces, axis=0)
        self.faces = np.unique(faces, axis=0)

    def __union_coords(self, coords):
        coords = np.append(self.coords, coords, axis=0)
        self.coords = np.unique(coords, axis=0)

    def __union_index(self, index):
        index = np.append(self.index, index, axis=0)
        self.index = np.unique(index, axis=0)

    def intersect(self, geometry):
        """
        Get intersection of two geometry classes.

        Parameters
        ----------
            geometry: an instance of Geometry class.
        """
        self.__intersect_coords(geometry.coords)
        self.__intersect_faces(geometry.faces)
        self.__intersect_index(geometry.index)

    def __intersect_coords(self, coords):
        result = []
        for i in self.coords:
            for j in coords:
                if np.all(i==j):
                    result.append(i)
        self.coords = np.array(result)

    def __intersect_faces(self, faces):
        result = []
        for i in self.faces:
            for j in faces:
                if np.all(np.in1d(i, j)):
                    result.append(i)
        self.faces = np.array(result)

    def __intersect_index(self, index):
        result = self.index[np.in1d(self.index, index)]
        self.index = result

    def exclude(self, geometry):
        """
        Exclude geometry out of self.

        Parameters
        ----------
            geometry: an instance of Geometry class.
        """
        self.__exclude_coords(geometry.coords)

    def __exclude_coords(self, coords):
        result = []
        for i in self.coords:
            match = 0
            for j in coords:
                if np.all(i == j):
                    match = 1
                    break
            if not match:
                result.append(i)
        self.coords = np.array(result)

    def __exclude_faces(self, faces):
        result = []
        for i in self.faces:
            match = 0
            for j in faces:
                if np.all(np.in1d(i, j)):
                    match = 1
                    break
            if not match:
                result.append(i)
        self.faces = np.array(result)

    def __exclude_index(self, index):
        result = [i for i in self.index if i not in index]
        self.index = result

    @property
    def centroid(self):
        """
        Get centroid of self geometry.

        Return
        ------
            cen: an instance of geometry class, centroid of self geometry.
        """
        # FIXME specify meaning of parameters.
        cen = Geometry(name=self.name)
        cen.coords = np.mean(self.coords, axis=0)
        cen.faces = None
        cen.index = None
        return cen


class Scalar(object):
    """
    Class of Scalar.

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
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) + other.get(idname))        
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __sub__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) - other.get(idname))
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __mul__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname) * other.get(idname))
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __div__(self, other):
        sa_ins = Scalar()
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
        

class Connection(object):
    def __init__(self, region=None, tract=None):
        """
        Init Connection.

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
            key: name of properties of Connection, should be one of ['coords', 'faces', index']

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
            key: name of properties of Connection, should be one of ['coords', 'faces', index']
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
        Merge another Connection class.

        Parameters
        ----------
            ca: an instance of Connection class.
        """
        self.region.append(ca.region)
        self.tract.append(ca.tract)

    def remove(self):
        pass

