
class Surface(object):
    """ Surface class represent brain surface data from surface analysis
        
        Attributes
        ----------
        mesh: Mesh object, brain surface 
        data: image data, a 3d or 4d array
        space: a string, native, mni152
        dims: image dimensions, a 3x1 or 4x1 array
        src: source of the image data, a string.
        """

    def __init__(self, mesh=None, data=None, space=None, src=None):
        self.mesh = mesh
        self.data = data
        self.space = space 
        self.src = src
    
    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, mesh):
        self.mesh = mesh
    
    @property
    def src(self):
        return self._src

    @src.setter
    def src(self,src):
        self._src = src

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

    def get_value(self, mask):
        """ Get the values of the voxels within the mask roi

        Parameters
        ----------
        mask

        Returns
        -------
        values: NxT numpy array, scalar value from the mask roi
        """
        pass
    
    def get_coords(self, mask):
        pass

    def load_mesh(self, filename):
        """ Update the mesh from a CIFIT file

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
        """ Save the mesh to a freesurfer format file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        
        """
        pass
    
    def load_data(self, filename):
        """ Update the data from a freesurfer scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: a Surface obejct
        """
        pass

    def save_data(self, filename):
        """ Save the data to a freesurfer scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """
        pass