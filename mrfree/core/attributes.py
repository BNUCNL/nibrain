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
            print('{} not exist.'.format(item))
            return None

    def __getitem__(self, item):
        return self.get(item)

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
        assert len(name) == data.shape[1], "Mismatch between name and data."
        if (self.name is None) | (self.data is None):
            self.name = name
            self.data = data
        else: 
            self.concatenate(name, data) 

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

    def concatenate(self, name = None, data = None):
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
            assert (self.name is not None) & (self.data is not None), "Please set name and data before concatenating it."
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
        

