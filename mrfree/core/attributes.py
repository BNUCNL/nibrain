#!/usr/bin/env python

# Attributes class

import numpy as np

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
        

