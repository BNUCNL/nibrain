#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np

class OperateCSV(object):
    """
    A class to help do operations on csv data
    """
    def __init__(self, data):
        """
        Initialize instance

        Parameters:
        -----------
        data: csv file data path or pd.DataFrame
        """
        if isinstance(data, str):
            assert datapath.endswith('csv'), "a .csv file should be inputed"
            self.rawdata = pd.read_csv(datapath)
        elif isinstance(data, pd.DataFrame):
            self.rawdata = data
        else:
            raise Exception('Please input a csv file name or a pandas DataFrame.')

    def getkeys(self, data = None):
        """
        Get all of the keys in rawdata

        Parameters
        ----------
        None 

        Returns:
        ---------
        data_keys: a list contains keys

        Examples:
        ---------
        >>> self.getkeys()
        """
        if data is None:
            data = self.rawdata
        return data.keys().tolist()

    def getdata_from_keys(self, keys, data = None):
        """
        Get data from a list of keys

        Parameters:
        -----------
        keys: a list of keys need to be extracted

        Return:
        --------
        pt_data: data as part of rawdata with specific series of keys

        Example:
        ---------
        >>> self.get_data_from_keys(['Subject', 'Gender'])
        """
        if data is None:
            data = self.rawdata
        return data.loc[:, keys]

    def get_data_by_row(self, source_key, source_value, data = None):
        """
        Get data by row
        
        Parameters:
        -----------
        source_key: get data that source_value is in source_key
        source_value: get specific row of data that contains source_value

        Return:
        --------
        pt_data: data that rows contains source_value in key of source_key

        Example:
        ---------
        >>> self.get_data_by_row('Subject', ['100004', '308331'])
        
        """
        if data is None:
            data = self.rawdata
        return data[data[source_key].isin(source_value)]

    def find_values_by_key(self, dest_key, source_key, source_value, data = None):
        """
        Find values in dest_key from source_value in source_key

        Parameters:
        -----------
        dest_key: destination key, a string. Values got from this column
        source_key: source key, a string.
        source_value: value as match condition to find values in destination key

        Returns:
        ---------
        match_data: data that satisfied match condition

        Example:
        --------
        >>> self.find_values_by_key('Subject', 'Family_ID', '23142_21345')
        """
        if data is None:
            data = self.rawdata
        return data[dest_key][data[source_key] == source_value]

    def unique_value_by_key(self, key, data = None):
        """
        Get unique value by a key

        Parameters:
        ------------
        key: a string. key name of data

        Return:
        -------
        unique_data: unique list data

        Examples:
        ---------
        >>> self.unique_value_by_key('Gender')
        """
        if data is None:
            data = self.rawdata
        return np.unique(data[key]).tolist()


