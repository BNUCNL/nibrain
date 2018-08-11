
class Surface(object):
    """ Surface class represent brain surface data from surface analysis
        
        Attributes
        ----------
        mesh: Mesh object, brain surface 
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        dims: image dimensions, a 3x1 or 4x1 array
        """

    def __init__(self, mesh=None, data=None, space=None):
        self.mesh = mesh
        self.data = data
        self.space = space 

    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, mesh):
        self.mesh = mesh

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
    
    def __add__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data + other.data
            
    def __sub__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data - other.data
        pass
    
    def __div__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data * other.data
        pass
    
    def __mul__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data/other.data

    def load_mesh(self, filename):
        """ Load mesh from surface mesh file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        Self: an Surface object
        """
        pass
    
    def save_mesh(self, filename):
        """ Save the mesh to a surface mesh file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        
        """
        pass
    
    def load_data(self, filename):
        """ Load the data from a surface scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a surface scalar file

        Returns
        -------
        self: a Surface obejct
        """
        pass

    def save_data(self, filename):
        """ Save the data to a surface scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a surface scalar file

        Returns
        -------

        """
        pass

    def get_roi_data(self, roi=None):
        """ Get the data of the vertex within a roi

        Parameters
        ----------
        roi, a roi object with the same type as
        if roi == None, return data from all vertices on the surface
        Returns
        -------
        values: NxT numpy array, scalar value from the mask roi
        """
        pass

    def get_roi_coords(self, roi=None):
        """ Get the coordinates of the vertex within a roi

        Parameters
        ----------
        roi

        Returns
        -------
        coords: Nx3 numpy array, scalar value from the roi
        """
        pass
