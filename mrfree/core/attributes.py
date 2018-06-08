#!/usr/bin/env python

# attributes class

import numpy as np
from mrfree.algorithms.geometry.geo_tools import faces_to_edges, faces_to_adjmatrix


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
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise ValueError('{} is not found.'.format(key))

    def set(self, key, value):
        """
        Set value to the property of Geometry by key.

        Parameters
        ----------
        key: name of properties of Geometry, should be one of ['coords', 'faces', index']
        value: value of properties that what to set.
        """
        if hasattr(self, key):
            self.__setattr__(key, value)
        else:
            raise ValueError('{} is not found.'.format(key))

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

    def adjmatrix(self, sym=True):
        """
        Get adjacency matrix of Geometry, calculated from faces.

        Parameters
        ----------
        sym: make adjmatrix symmetrical, default is True.

        Returns
        -------
        adjm: adjacency matrix of (subj_id, hemi, surf), if mask=None, then shape = (n_vertexes, n_vertexes).
        """
        adjm = faces_to_adjmatrix(self.faces, sym=sym)
        return adjm

    def get_edges(self):
        """
        Get edges of Geometry, calculated from faces.

        Returns
        -------
        edges: array, edges of brain surface mesh, shape=(n_edges, 2).
        """
        edges = faces_to_edges(self.faces)
        return edges


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
            pass

    def __setattr__(self, item, value):
        object.__setattr__(self, item, value)

    def __getattr__(self, item):
        if item not in self.__dict__:
            return None

    def __getitem__(self, item):
        return self.get(item)[1]

    def __add__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname)[1] + other.get(idname)[1])        
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __sub__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname)[1] - other.get(idname)[1])
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __mul__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, self.get(idname)[1] * other.get(idname)[1])
            except ValueError:
                raise Exception('{} mismatched'.format(idname))
        return sa_ins

    def __div__(self, other):
        sa_ins = Scalar()
        same_indices = np.unique([i for i in self.name if i in other.name])
        for i, idname in enumerate(same_indices):
            try:
                sa_ins.set(idname, 1.0*self.get(idname)[1] / other.get(idname)[1])
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
        if data.ndim == 1:
            data = data[...,np.newaxis]
        if (len(np.unique(name)) == 1)&(len(name)<data.shape[1]):
            name = [name[0]]*data.shape[1]
        else:
            assert len(name)==data.shape[1], "Mismatch between name and data." 
        self.name = name
        self.data = data

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
        if isinstance(name, str):
            name = [name]
        if self.name is not None:
            # find all occurrences of name in a list of self.name
            indices = [i for i, x in enumerate(self.name) if x in name]

            if len(self.name) == len(name):
                assert len(indices) == len(name), "Exist mismatched feature(s)."
            else:
                assert len(np.unique(self.name)) == len(np.unique(self.name)), "Exist mismatched feature(s)."

            if len(indices) == 0:
                print('Name mismatched.')
                return None
            else:
                dataidx = [self.name[i] for i in indices]
                sorted_dataidx = sorted(dataidx)
                sorted_data = self.data[:, np.argsort(dataidx)]
                return sorted_dataidx, sorted_data
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
        if isinstance(name, str):
            name = [name]
        for na in name:
            assert na in self.name, "Name mismatched."
            indices = [i for i, x in enumerate(self.name) if x == na]
            self.name = [x for i, x in enumerate(self.name) if i not in indices]
            self.data = np.delete(self.data, indices, axis=1)

    def sort(self, reverse = False):
        """
        Sorted scalar instance by features

        Args:
            reverse: sorting order.
                     By default is False, in ascending order.
                     if True, by descending order.
        """
        self.name = sorted(self.name, reverse = reverse)
        if reverse is False:
            self.data = self.data[:, np.argsort(self.name)]
        else:
            self.data = self.data[:, np.argsort(self.name)[::-1]] 

    def aggregate(self, scalar, feature = None):
        """
        Aggregate data in a new scalar to a existed scalar.

        Args:
            scalar: scalar instance
            feature: feature (identity) list.
                     Select specific feature(s) to aggregate with.

        Returns:
            A new scalar instance that has been aggregating. 
        """
        sa_ins = Scalar()
        if feature is None:
            assert sorted(self.name) == sorted(scalar.name), "Feature mismatched."          
            name = sorted(self.name)
            agg_data = np.vstack((self.data[:,np.argsort(self.name)], scalar.data[:,np.argsort(scalar.name)]))      
        else: 
            if isinstance(feature, str):
                feature = [feature]
            name1, data1 = self.get(feature) 
            name2, data2 = scalar.get(feature)
            assert name1 == name2, "Existing mismatched feature."
            name = name1
            agg_data = np.vstack((data1, data2))
        sa_ins.set(name, agg_data)
        return sa_ins
            
    def add(self, scalar, feature = None):
        """
        Scalar addition

        Args:
            scalar: scalar instance
            feature: feature (identity) list.
                     Select specific feature(s) for operation

        Returns:
            A new scalar instance contains data from addition of data in two original scalar instance.
        """
        sa_ins = Scalar()
        if feature is None:
            assert sorted(self.name) == sorted(scalar.name), "Feature mismatched."
            name = self.name
            add_data = self.data[:,np.argsort(self.name)] + scalar.data[:, np.argsort(self.name)]
        else:
            if isinstance(feature, str):
                feature = [feature]
            name1, data1 = self.get(feature)
            name2, data2 = scalar.get(feature)
            assert name1 == name2, "Existing mismatched feature."
            name = name1
            add_data = data1 + data2
        sa_ins.set(name, add_data)
        return sa_ins

    def subtract(self, scalar, feature = None):
        """
        Scalar subtraction

        Args:
            scalar: scalar instance
            feature: feature (identity) list.
                     select specific feature(s) for operation.

        Returns:
            A new scalar instance contains data from subtraction of data in two orginal scalar instance.
        """
        sa_ins = Scalar()
        if feature is None:
            assert sorted(self.name) == sorted(scalar.name), "Feature mismatched."
            name = self.name
            subtract_data = self.data[:, np.argsort(self.name)] - scalar.data[:, np.argsort(self.name)]
        else:
            if isinstance(feature, str):
                feature = [feature]
            name1, data1 = self.get(feature)
            name2, data2 = scalar.get(feature)
            assert name1 == name2, "Existing mismatched feature."
            name = name1
            subtract_data = data1 - data2
        sa_ins.set(name, subtract_data)
        return sa_ins 

    def multiply(self, scalar, feature = None):
        """
        Scalar multiply.

        Args: 
            scalar: scalar instance
            feature: feature (identity) list.
                     Select specific feature(s) to operation.

        Returns:
            A new scalar instance contains data from multiplication of data in two original scalar instance.
        """ 
        sa_ins = Scalar()
        if feature is None:
            assert sorted(self.name) == sorted(scalar.name), "Feature mismatched."
            name = self.name
            multiply_data = self.data[:, np.argsort(self.name)] * scalar.data[:, np.argsort(self.name)]
        else:
            if isinstance(feature, str):
                feature = [feature]
            name1, data1 = self.get(feature)
            name2, data2 = self.get(feature)
            assert name1 == name2, "Existing mismatched feature."
            name = name1
            multiply_data = data1 * data2
        sa_ins.set(name, multiply_data)
        return sa_ins

    def divide(self, scalar, feature = None):
        """
        Scalar division

        Args:
            scalar: scalar instance
            feature: feature (identity) list.
                     Select specific feature(s) to operation.

        Returns:
            A new scalar instance contains data from division of data in two original scalar instance.
        """
        sa_ins = Scalar()
        if feature is None:
            assert sorted(self.name) == sorted(scalar.name), "Feature mismatched."
            name = self.name
            div_data = self.data[:, np.argsort(self.name)] + scalar.data[:, np.argsort(self.name)]
        else:
            if isinstance(feature, str):
                feature = [feature]
            name1, data1 = self.get(feature)
            name2, data2 = scalar.get(feature)
            assert name1 == name2, "Existing mismatched feature."
            name = name1
            div_data = 1.0 * data1 / data2
        sa_ins.set(name, div_data)
        return sa_ins


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

