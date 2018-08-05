#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from scipy.stats import zscore

class Scalar(object):
    """
     Class to pack scalar 2d data. The implementation is based on the pands DataFrame class

    Attributes:
        index : Index or array-like. Row labels to use for resulting frame.
        columns : Index or array-like.Column labels to use for resulting frame.
        data: a pands DataFrame to contain scalar data
    """

    def __init__(self, data=None, index=None, columns=None):
        
        """
        Initialize the data as a pands DataFrame object
        
        Parameters
        ----------
        data : numpy ndarray (structured or homogeneous), dict, or DataFrame. 
        Dict can contain Series, arrays, constants, or list-like objects
        
        index : Index or array-like. 
        Row labels to use for resulting frame. Will default to RangeIndex if no indexing 
        information part of input data and no index provided
        
        columns : Index or array-like.
        Column labels to use for resulting frame. Will default to RangeIndex (0, 1, 2, ¡­, n) 
        if no column labels are provided
        """

        self.data = pd.DataFrame(data, index, columns)
        self.index = self.data.index
        self.columns = self.data.columns
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    def __getitem__(self, item):
        index, columns = item
        self.data.iloc(index, columns)

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns
    
    @property
    def values(self):
        return self.data.values

    def loc(self, index=None, columns=None):
        """ Access a group of rows and columns by label(s) or a boolean array.
        .loc[] is primarily label based, but may also be used with a boolean array.
        
        
        Parameters
        ----------
        index : Index or array-like. Row labels to use for resulting frame. Allowed inputs are: 
        A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and 
        never as an integer position along the index).
        A list or array of labels, e.g. ['a', 'b', 'c'].
        A slice object with labels, e.g. 'a':'f'.
        A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
        A callable function with one argument (the calling Series, DataFrame or Panel) and 
        that returns valid output for indexing (one of the above)
               
        columns : Index or  Column labels to use for resulting frame. Allowed inputs are the same as index 
        
        Return
        ------------
        a DataFrame Object
        
        """
        return self.data.loc[index, columns]

    def iloc(self,index=None, columns=None):
        """Purely integer-location based indexing for selection by position.
        .iloc[] is primarily integer position based (from 0 to length-1 of the axis), 
        but may also be used with a boolean array.
        
        Parameters
        ----------
        index: array-like. Allowed inputs are:
                 An integer, e.g. 5.
                 A list or array of integers, e.g. [4, 3, 0].
                 A slice object with ints, e.g. 1:7.
                 A boolean array.
         
        columns: array-like. Allowed inputs are the same as index.
        
        Return
        ----------
        a DataFrame object

        """
        
        return self.data.iloc[index, columns]

    def add(self, other, axis='columns', level=None, fill_value=None):
        """Addition of dataframe and other, element-wise (binary operator add).
        
           Parameters
           ---------- 
            other : Series, DataFrame, or constant
            axis : {0, 1, 'index', 'columns'}
            level : int or name, 
            Broadcast across a level, matching Index values on the passed MultiIndex level
            fill_value : None or float value, default None
            Fill existing missing (NaN) values, and any new element needed for successful DataFrame alignment, 
            with this value before computation. If data in both corresponding DataFrame locations is missing 
            the result will be missing
            
           Return
           ----------
           a DataFrame object

        """
        self.data.add(other, axis, level, fill_value)
        return self

    def sub(self, other, axis='columns', level=None, fill_value=None):
        """Subtraction of dataframe and other, element-wise (binary operator sub).
        
           Parameters
           ---------- 
            other : Series, DataFrame, or constant
            axis : {0, 1, 'index', 'columns'}
            level : int or name, 
            Broadcast across a level, matching Index values on the passed MultiIndex level
            fill_value : None or float value, default None
            Fill existing missing (NaN) values, and any new element needed for successful DataFrame alignment, 
            with this value before computation. If data in both corresponding DataFrame locations is missing 
            the result will be missing

           Return
           ----------
           a DataFrame object

        """
        
        self.data.sub(other,axis, level, fill_value)
        return self

    def mul(self, other, axis='columns', level=None, fill_value=None):
        """Multiplication of dataframe and other, element-wise (binary operator mul).

           Parameters
           ---------- 
            other : Series, DataFrame, or constant
            axis : {0, 1, 'index', 'columns'}
            level : int or name, 
            Broadcast across a level, matching Index values on the passed MultiIndex level
            fill_value : None or float value, default None
            Fill existing missing (NaN) values, and any new element needed for successful DataFrame alignment, 
            with this value before computation. If data in both corresponding DataFrame locations is missing 
            the result will be missing

           Return
           ----------
           a DataFrame object

        """
        self.data.mul(other.data, axis, level, fill_value)
        return self

    def div(self, other, axis='columns', level=None, fill_value=None):
        """Floating division of dataframe and other, element-wise (binary operator truediv).

           Parameters
           ---------- 
            other : Series, DataFrame, or constant
            axis : {0, 1, 'index', 'columns'}
            level : int or name, 
            Broadcast across a level, matching Index values on the passed MultiIndex level
            fill_value : None or float value, default None
            Fill existing missing (NaN) values, and any new element needed for successful DataFrame alignment, 
            with this value before computation. If data in both corresponding DataFrame locations is missing 
            the result will be missing

           Return
           ----------
           a DataFrame object

        """
        self.data.div(other.data,axis, level, fill_value)
        return  self

    def abs(self):
        """Return a Series/DataFrame with absolute numeric value of each element.
        
         Return
         ----------
         Series/DataFrame containing the absolute value of each element.

        """
        return self.data.abs()

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None):
        """Return the mean of the values for the requested axis

        Parameters
        ----------
        axis : {index (0), columns (1)}
        skipna : boolean, default True
        Exclude NA/null values when computing the result.
        level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series
        numeric_only : boolean, default None
        Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
        
        Returns
        ----------	
        mean : Series or DataFrame (if level specified)
        """
        return self.data.mean(axis, skipna, level, numeric_only)

    def std(self, axis=None, skipna=None, level=None, numeric_only=None):
        """Return sample standard deviation over requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
        skipna : boolean, default True
        Exclude NA/null values when computing the result.
        level : int or level name, default None
        If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series
        numeric_only : boolean, default None
        Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.

        Returns
        ----------	
        std : Series or DataFrame (if level specified)
        """
        return self.data.std(axis, skipna, level, numeric_only)

    def zscore(self):
        self.data.apply(zscore)

    @property
    def shape(self):
        """
        Return
        ---------------
        Return a tuple representing the dimensionality of the DataFrame. 
        """
        return self.data.shape

    @property
    def ndim(self):
        """
        Return
        -------------- 
        Return an int representing the number of axes / array dimensions.

        """
        return self.data.ndim

    def append(self, other, ignore_index=False, verify_integrity=False):
        """Append rows of other to the end of this frame, returning a new object. Columns not in this frame are added as new columns.
        Parmeters:	
        other : DataFrame or Series/dict-like object, or list of these
        The data to append.
        ignore_index : boolean, default False
        If True, do not use the index labels.
        verify_integrity : boolean, default False
        If True, raise ValueError on creating index with duplicates.
        
        Returns
        -----------	
        appended : DataFrame

        """
        self.data.append(other.data, ignore_index, verify_integrity)
        return self

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        self.data.join(other, on, how, lsuffix, rsuffix, sort)

    def remove(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """Drop specified labels from rows or columns.
        Remove rows or columns by specifying label names and corresponding axis, or by specifying directly index or column names. 
        When using a multi-index, labels on different levels can be removed by specifying the level.
        
        Parameters
        --------------	
        labels : single label or list-like
        Index or column labels to drop.
        axis : {0 or 'index', 1 or 'columns'}, default 0
        Whether to drop labels from the index (0 or 'index') or columns (1 or 'columns').
        index, columns : single label or list-like
        Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
        New in version 0.21.0.
        level : int or level name, optional
        For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
        If True, do operation inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise¡
        If 'ignore', suppress error and only existing labels are dropped.
        
        Returns
        --------------	
        dropped : pandas.DataFrame
        
        Raises
        --------------
        KeyError, If none of the labels are found in the selected axis
        """
        self.data.drop(labels, axis, index, columns, level, inplace, errors)
        return self

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind='quicksort',
                   na_position='last', sort_remaining=True, by=None):
        """Sort object by labels (along an axis)
        
            Parameters:
            ---------------
            axis : index, columns to direct sorting
            level : int or level name or list of ints or list of level names
            if not None, sort on values in specified index level(s)
            ascending : boolean, default True
            Sort ascending vs. descending
            inplace : bool, default False
            if True, perform operation in-place
            kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
            Choice of sorting algorithm. See also ndarray.np.sort for more information. mergesort is the only stable algorithm.
            For DataFrames, this option is only applied when sorting on a single column or label.
            na_position : {'first', 'last'}, default 'last'
            first puts NaNs at the beginning, last puts NaNs at the end. Not implemented for MultiIndex.
            sort_remaining : bool, default True
            if true and sorting by level and index is multilevel, sort by other levels too (in order) after sorting by specified level

            Returns
            --------------
            sorted_obj : DataFrame

         """

        self.data.sort_index(axis, level, ascending, inplace, kind,na_position, sort_remaining, by)
        return self

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):

      """Sort by the values along either axis
      
        Parameters
        --------------
        by : str or list of str
        Name or list of names to sort by.
        if axis is 0 or 'index' then by may contain index levels and/or column labels
        if axis is 1 or 'columns' then by may contain column levels and/or index labels
        Changed in version 0.23.0: Allow specifying index or column level names.
        axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis to be sorted
        ascending : bool or list of bool, default True
        Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.
        inplace : bool, default False
        if True, perform operation in-place
        kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
        Choice of sorting algorithm. See also ndarray.np.sort for more information. mergesort is the only stable algorithm. 
        For DataFrames, this option is only applied when sorting on a single column or label.
        na_position : {'first', 'last'}, default 'last'
        first puts NaNs at the beginning, last puts NaNs at the end
        
        Returns
        ----------------	
        sorted_obj : DataFrame

        """

      self.data.sort_value(by, axis, ascending, inplace, kind, na_position)
      return self

    def read_from_cifti(self):
        """ Construct Scalar object by reading a CIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: a Lines object
        """

        pass

    def save_to_cifti(self, filename):
        """ Save Scalar object to a CIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """
        pass

    def read_from_nifti(self, filename):
        """ Construct Scalar object by reading a NIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a NIFTI file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_to_nifti(self, filename):
        """ Save Scalar object to a NIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a NIFTI file

        Returns
        -------
       """
        pass


    def read_from_gifti(self, filename):
        """ Construct Scalar object by reading a GIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_to_gifti(self, filename):
        """ Save Scalar object to a GIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------
       """
        pass