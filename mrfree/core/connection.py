

import  region, tract

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

