#!/usr/bin/env python

import numpy as np
from base import intersect2d,exclude2d

class Points(object):
    """Points represent a collection of spatial ponits
    
    Attributes
    ----------
    data:  Nx3 numpy array, points coordinates
    id: Nx1 numpy array, id for each point
    src: str, source of the points data
    """
    def __init__(self, data=None, id=None, src=None):
        """
        Parameters
        ----------
        data:  Nx3 numpy array, points coordinates
        id: Nx1 numpy array, id for each point
        src: str, source of the points data
        """
        self.data  = data
        self.id = id
        self.src = src

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        assert data.ndim == 2 and data.shape[1] == 3, "data should be N x 3 np array."
        self._data = data

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        self._src = src

    def merge(self, other):
        """ Merge other Points object into self
        
        Parameters
        ----------
        other: Points object to be merged

        Returns
        -------
        self: the merged object

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = np.vstack(self.data)
        self.data = np.unique(self.data,axis=0)
        return self

    def intersect(self,other):
        """ Intersect with other Points object

        Parameters
        ----------
        other: Points object to be intersectd with this object

        Returns
        -------
        self: the intersected object

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = intersect2d(self.data, other.data)
        return self

    def exclude(self, other):
        """ Exclude other Points object from this object

        Parameters
        ----------
        other: Points object to be merged

        Returns
        -------
        self: the object after excluding others

        """
        assert isinstance(other, Points), "other should be a Points object"
        self.data = exclude2d(self.data, other.data)
        return self

    def get_center(self):
        """ Get the center of the points set
        
        Parameters
        ----------

        Returns
        -------
        center: 1x3 numpy array, the center coordinates

        """
        
        return np.mean(self.data,axis=0)

    def update_from_cifti(self):
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
        """ Save Points object to a CIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """
        pass

    def update_from_nifti(self, filename):
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
        """ Save Points object to a NIFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a NIFTI file

        Returns
        -------
       """
        pass

    def update_from_gifti(self, filename):
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
        """ Save Points object to a GFTI file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------
        """
        pass


class Lines(object):
    def __init__(self,source):
        """
        Parameters
        ----------
        data: geometry data, a sequence of array.
        id: the id for each array.
        source: source of the geometry data, a string.
        """
        assert isinstance(source, str), "Input 'source' should be a string of tck file path."
        self._src = []
        self._src.append(source)
        self._data = None
        self._id = None
        #Reason for changed:the way of 'self.name' can be modified arbitrarily and doesn't need @property.

    @property
    def src(self):
        return self._src
    #Reason for changed:source shouldn't be modified arbitrarily.

    @property
    def data(self):
        return self._data
    #Reason for changed:data can read from load source file.

    @property
    def id(self):
        return self._id
    # Reason for changed:id can acquire from the data.

    def merge(self, other_lines):
        """ Merge other tract into the Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another lines
        axis: integer, 0 or 1

        Return
        ----------
        self: merged Lines
        """
        assert isinstance(other, Lines), "other should be a Lines object"
        self._src.append(other_lines.src)
        other_lines_id = [i+len(self._id) for i in other_lines.id]
        self._id.extend(other_lines_id)
        #self._data.extend(other_lines_data)
        #A arraysequence method can be used for this situation.

    def intersect(self,other):
        """ Intersect with other Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another Lines 

        Return
        ----------
        self with intersection from two Lines
        """
        pass

    def exclude(self, other):
        """ Exclude other Lines from the current Lines based on the line id.

        Parameters
        ----------
        other: Lines object, another Lines 

        Return
        ----------
        self:  The Lines after excluding other Lines
        """
        pass

    def equidistant_resample(self, num_segment):
        """ Resample the Lines with equidistantance
        
        Parameters
        ----------
        num_segment: int, number of segment to be sampled

        Returns
        -------
        
        self: a resampled Lines

        """
        pass

    def skeleton(self):
        """Find the skeletion of the line set
        
        Returns
        -------
        skeleton: a Line object

        """
        pass
      
    def update_from_tck(self, filename):
        """ Construct Lines object by reading a TCK file
    
        Parameters
        ----------
        filename: str
            Pathstr to a TCK file

        Returns
        -------
        self: a Lines object
        """
        self._tract = streamlines.TckFile.load(self._src)
        self._data = self._tract.streamlines
        self._id = list(range(len(self._data)))

    def save_to_tck(self, filename):
        """ Save Lines object to a TCK file

        Parameters
        ----------
        filename: str
            Pathstr to a TCK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def update_from_trk(self, filename):
        """ Construct Lines object by reading a TRK file

        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_to_trk(self, filename):
        """ Save Lines object to a TRK file

        Parameters
        ----------
        filename: str
            Pathstr to a TRK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def update_from_vtk(self, filename):
        """ Construct Lines object by reading a VTK file

        Parameters
        ----------
        filename: str
            Pathstr to a VTK file

        Returns
        -------
        self: a Lines object
        """
        pass

    def save_to_vtk(self, filename):
        """ Save Lines object to a VTK file

        Parameters
        ----------
        filename: str
            Pathstr to a VTK file

        Returns
        -------
        self: a Lines object
        """
        pass


class Mesh(object):
    """Mesh class represents geometry mesh
    
        Attributes
        ----------
        vertices
        faces
        edges
    
    """
    
    def __init__(self, vertices, faces, edges):
        self.vertices = vertices
        self.faces = faces
        self.edges = edges
    
    def __eq__(self, other):
        return  np.array_equal(self.vertices, self.vertices)
    
    
    def update_from_freesurfer(self):
        pass 
    
    
    def update_from_gifti(self, filename):
        """ Construct Lines object by reading a TCK file

        Parameters
        ----------
        filename: str
            Pathstr to a TCK file

        Returns
        -------
        self: a Lines object
        """
        pass
    

if __name__ == "__main__":
    # Test Points
    data = np.random.rand(10,3)
    id = 1
    src = "Faked points"
    rg = Points(data, id, src)
    rg.data = np.random.rand(5,3)
    rg.id = 2
    rg.src = "New faked points"

    # Test Lines
    data = [np.array([[0, 0., 0.9],
                      [1.9, 0., 0.]]),
            np.array([[0.1, 0., 0],
                      [0, 1., 1.],
                      [0, 2., 2.]]),
            np.array([[2, 2, 2],
                      [3, 3, 3]])]
    id = np.arange(len(data))
    src = "Faked lines"
    lg = Lines(data, id, src)
    lg.data =  rg.data.remove(1)
    lg.id = np.delete(rg.id,1)
    lg.src = "New faked lines"


