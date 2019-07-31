import os
import numpy as np
from geometry import Mesh
import nibabel as nib
from nibabel import freesurfer
from nibabel.spatialimages import ImageFileError


class Surface(object):
    """ Surface class represent brain surface data from surface analysis
        
        Attributes
        ----------
        mesh: Mesh object or surface obejct from nibabel, brain surface
        data: scalar data on the mesh, a 2d array
        space: a string, native, mni152
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
        self._mesh = mesh

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
    
    def __add__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data + other.data
        return  self
            
    def __sub__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data - other.data
        return self
    
    def __div__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data * other.data
        return self
    
    def __mul__(self, other):
        if self.mesh == other.mesh:
            self.data = self.data/other.data
        return self

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
        if filename.endswith(('.inflated', '.white', '.pial', '.orig')):
            vertices, faces = freesurfer.read_geometry(filename)

        elif filename.endswith('.surf.gii'):
            geo_img = nib.load(filename)
            vertices = geo_img.darray[0].data
            faces = geo_img.darray[1].data

        else:
            suffix = os.path.split(filename)[1].split('.')[-1]
            raise ImageFileError('The format-{} is not supported at present.'.format(suffix))

        self.mesh = Mesh(vertices, faces)
    
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

        if filename.endswith(('.curv', '.sulc', '.volume', '.thickness', '.area')):
            data = np.expand_dims(freesurfer.read_morph_data(filename), axis=-1)

        elif filename.endswith(('.shape.gii', '.func.gii')):
            data = np.expand_dims(nib.load(filename).darrays[0].data, axis=-1)

        elif filename.endswith(('.mgz', '.mgh')):
            data = nib.load(filename).get_data()
            data = data.reshape((data.shape[0], data.shape[-1]))

        elif filename.endswith(('.dscalar.nii', '.dseries.nii')):
            data = nib.load(filename).get_data()
            data = data.T

        elif filename.endswith('.label.gii'):
            data = np.expand_dims(nib.load(filename).darrays[0].data, axis=-1)

        elif filename.endswith('.dlabel.nii'):
            data = nib.load(filename).get_data().T

        elif filename.endswith('.label'):
            data = np.expand_dims(freesurfer.read_label(filename), axis=-1)

        elif filename.endswith('.annot'):
            data, _, _ = freesurfer.read_annot(filename)

        else:
            suffix = os.path.split(filename)[1].split('.')[-1]
            raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

        self.data = data

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
        roi: Nx1 numpy array, tuple, list, vertex id in the roi
        if roi is None, return data from all vertices on the surface

        Returns
        -------
        data: NxT numpy array, scalar value from the mask roi
        """
        if roi is not None:
            #roi = np.asarray(roi)
            #roi = roi.astype(int)
            data = self.data[roi, :]
        else:
            data = self.data

        return data

    def get_roi_coords(self, roi=None):
        """ Get the coordinates of the vertex within a roi

        Parameters
        ----------
        roi: numpy array or a tuple,list, vertex id in the roi
        if roi is None, return data from all vertices on the surface

        Returns
        -------
        coords: Nx3 numpy array, scalar value from the roi
        """
        if roi is not None:
            #roi = np.asarray(roi)
            #roi = roi.astype(int)
            coords = self.mesh.vertices[roi, :]
        else:
            coords = self.mesh.vertices

        return coords