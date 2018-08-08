
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
    

    def update_mesh_from_freesurfer(self, filename):
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
    
    def save_mesh_to_freesurfer(self, filename):
        """ Save the mesh to a freesurfer format file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        
        """
        pass
    
    def update_data_from_freesurfer(self, filename):
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

    def save_data_to_freesurfer(self, filename):
        """ Save the data to a freesurfer scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------

        """
        pass
    

    def update_mesh_from_gifti(self, filename):
        """ Update the mesh a GIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------
        self: an Surface obejct
        """
        pass
    
    def save_mesh_to_gifti(self, filename):
        """ Save the mesh to a gifti surface file

        Parameters
        ----------
        filename: str
            Pathstr to a GIFTI file

        Returns
        -------

        """
        pass

    def update_data_from_gifti(self, filename):
        """ Update the data from a GIFIT scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: an Surface obejct

        """
        pass
    
    def save_data_to_gifti(self, filename):
        """ Save the data to a GIFIT scalar file

        Parameters
        ----------
        filename: str
            Pathstr to a scalar GIFTI file

        Returns
        -------

        """
        pass

    def update_data_from_cifti(self, filename):
        """ Read image from a CIFIT file

        Parameters
        ----------
        filename: str
            Pathstr to a CIFTI file

        Returns
        -------
        self: an Surface obejct
        """
        pass