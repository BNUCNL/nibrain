#!/usr/bin/env python

import pandas as pd
from scipy.stats import zscore


"""We use pands DataFrame to implement our scalar"""



class Scalar(object):
    """
    Class of Scalar.
    Attributes:
        name: A string or list as identity of scalar data.
        data: Scalar data.
    """

    def __init__(self, data=None, index=None, columns=None):
        """
        Initialize the data as a pands DataFrame object
        """
        self.index = index
        self.columns = columns
        self.data = pd.DataFrame(data, index, columns)

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    def loc(self,index=None, columns=None):
        return self.data.loc[index, columns]

    def iloc(self,index=None, columns=None):
        return self.data.iloc[index, columns]

    def add(self, other, axis='columns', level=None, fill_value=None):
        self.data.add(other,axis, level, fill_value)
        return self

    def sub(self, other, axis='columns', level=None, fill_value=None):
        self.data.sub(other,axis, level, fill_value)
        return self

    def mul(self, other, axis='columns', level=None, fill_value=None):
        self.data.mul(other.data, axis, level, fill_value)
        return self

    def div(self, other, axis='columns', level=None, fill_value=None):
        self.data.div(other.data,axis, level, fill_value)
        return  self

    def abs(self):
        return self.data.abs()

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None):
        return self.data.mean(axis, skipna, level, numeric_only)

    def std(self, axis=None, skipna=None, level=None, numeric_only=None):
        return self.data.std(axis, skipna, level, numeric_only)

    def zscore(self):
        self.data.apply(zscore)

    def shape(self):
        return self.data.shape

    def ndim(self):
        return self.data.ndim

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        self.data.append(other.data, ignore_index, verify_integrity, sort)
        return self

    def remove(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """
        A method to delete scalar data which stored in the object.
        Args:
            name: Identity of data.
        """
        self.data.drop(labels, axis, index, columns, level, inplace, errors)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
                   na_position='last', sort_remaining=True, by=None):
        self.data.sort_index(axis, level, ascending, inplace, kind,na_position, sort_remaining, by)
        return self

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        self.data.sort_value(by, axis, ascending, inplace, kind, na_position)
        return self




