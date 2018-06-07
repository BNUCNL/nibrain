# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import scipy
import numpy as np
from math import pi
import nibabel as nib
import scipy.ndimage as ndimagei
from scipy import sparse
conn_dict = {6:1, 18:2, 26:3}

def is_in_image(v, shape):
    """
    
    Contributions
    -------------
        Author: 
        Editor: 
    
    """
    
    if np.rank(v) == 1:
        return ((v[0] >= 0) & (v[0] < shape[0]) &
                (v[1] >= 0) & (v[1] < shape[1]) &
                (v[2] >= 0) & (v[2] < shape[2]))
    else:
        return np.all([v[:,0] >= 0,v[:,0] < shape[0],
                    v[:,1] >= 0,v[:,1] < shape[1],
                    v[:,2] >= 0,v[:,2] < shape[2]],axis = 0)

class neighbor:
    """Define neighor for pixel or voxel.
    
    Return
    ------
        offsets: 2xN or 3xN np array
    
    Contributions
    -------------
        Author: 
        Editor: 
    
    """
    def __init__(self, nbdim, nbsize, res=[1,1,1]):
        self.nbdim =  nbdim
        self.nbsize = nbsize
        self.res = res

    def offsets(self):
        return self.compute_offsets()

class pixelconn(neighbor):
    """Define pixel connectivity for 2D or 3D image.
    
    Returns
    -------
        offsets: 2 x N or 3 x N np array, 
                N = nbdim + 1(current pixel is included)
    
    Contributions
    -------------
        Author: 
        Editor: 
    
    """

    def compute_offsets(self):
        if self.nbdim == 2: # 2D image 4, 6, 8-connected 
            if self.nbsize == 4:    
                offsets = np.array([[0, 0],
                                    [1, 0],[-1, 0],
                                    [0, 1],[0, -1]])
            elif self.nbsize == 6:                 
                offsets = np.array([[0, 0],
                                    [1, 0],[-1, 0],
                                    [0, 1],[0, -1],
                                    [1, 1], [-1, -1]])
            elif self.nbsize == 8: 
                offsets = np.array([[0, 0],
                                    [1, 0],[-1, 0],
                                    [0, 1],[0, -1],
                                    [1, 1], [-1, -1]
                                    [1, -1], [-1, 1]])
        elif self.nbdim == 3: # 3D volume 6, 18, 26-connected
            if self.nbsize == 6: 
                offsets = np.array([[0, 0, 0],
                                    [1, 0, 0],[-1, 0, 0],
                                    [0, 1, 0],[0, -1, 0],
                                    [0, 0, -1], [0, 0, -1]])      
            elif self.nbsize == 18: 
                offsets = np.array([[0, 0, 0],
                                    [0,-1,-1],[-1, 0,-1],[0, 0,-1],
                                    [1, 0,-1],[0, 1,-1],[-1,-1, 0],
                                    [0,-1, 0],[1,-1, 0],[-1, 0, 0],
                                    [1, 0, 0],[-1, 1, 0],[0, 1, 0],
                                    [1, 1, 0],[0,-1, 1],[-1, 0, 1],
                                    [0, 0, 1],[1, 0, 1],[0, 1, 1]])
        
            elif self.nbsize == 26: 
                offsets = np.array([[0, 0, 0],
                                    [-1,-1,-1],[0,-1,-1],[1,-1,-1],
                                    [-1, 0,-1],[0, 0,-1],[1, 0,-1],
                                    [-1, 1,-1],[0, 1,-1],[1, 1,-1],
                                    [-1,-1, 0],[0,-1, 0],[1,-1, 0], 
                                    [-1, 0, 0],[1, 0, 0],[-1, 1, 0],
                                    [0, 1, 0],[1, 1, 0],[-1,-1, 1],
                                    [0,-1, 1],[1,-1, 1],[-1, 0, 1],
                                    [0, 0, 1],[1, 0, 1],[-1, 1, 1],
                                    [0, 1, 1],[1, 1, 1]])
        return offsets.T


class sphere(neighbor):
    """Sphere neighbor for pixel or voxel.
    
    Contributions
    -------------
        Author: 
        Editor: 
    
    """
    
    def compute_offsets(self):
        offsets = []
        if self.nbdim == 2:
            nbsizex = int(self.nbsize/self.res[0])
            nbsizey = int(self.nbsize/self.res[1])
            for x in np.arange(-nbsizex, nbsizex + 1):
                for y in np.arange(-nbsizey, nbsizey + 1):
                    if np.linalg.norm([x*res[0],y*res[1]]) <= self.nbsize:
                            offsets.append([x,y])
        elif self.nbdim == 3:
            nbsizex = int(self.nbsize/self.res[0])
            nbsizey = int(self.nbsize/self.res[1])
            nbsizez = int(self.nbsize/self.res[2])
            for x in np.arange(-nbsizex, nbsizex + 1):
                for y in np.arange(-nbsizey, nbsizey + 1):
                    for z in np.arange(-nbsizez, nbsizez + 1):
                        if np.linalg.norm([x*self.res[0],y*self.res[1],z*self.res[2]]) <= self.nbsize:
                            offsets.append([x,y,z])
        else: print 'wrong nbdim'

        return np.array(offsets).T
     
class cube(neighbor):
    """
    
    Contributions
    -------------
        Author: 
        Editor: 
    
    """
    
    def compute_offsets(self):
        offsets = []
        if self.nbdim == 2:
            nbsizex = int(self.nbsize/self.res[0])
            nbsizey = int(self.nbsize/self.res[1])
            for x in np.arange(-nbsizex, nbsizex + 1):
                for y in np.arange(-nbsizey, nbsizey + 1):
                    offsets.append([x,y])
        elif self.nbdim == 3:
            nbsizex = int(self.nbsize/self.res[0])
            nbsizey = int(self.nbsize/self.res[1])
            nbsizez = int(self.nbsize/self.res[2])
            for x in np.arange(-nbsizex, nbsizex + 1):
                for y in np.arange(-nbsizey, nbsizey + 1):
                    for z in np.arange(-nbsizez, nbsizez + 1):
                        offsets.append([x,y,z])
        
        else: print 'wrong nbdim'
        return np.array(offsets).T

                    
class  reho_volneighbors(pixelconn):
    
    def __init__(self, imgdat, nbdim=3, nbsize=26, res=[1,1,1], shape='fast_cube'):
        self.data = imgdat
        self.data1d = imgdat.flatten()
        self.imgidx1d = np.array(np.nonzero(self.data1d > 0)).T
        self.voxnum = np.shape(self.imgidx1d)[0]

        if shape == 'fast_cube':
            self.nb_shape = pixelconn(nbdim, nbsize)
        elif shape == 'sphere':
            self.nb_shape = sphere(nbdim, nbsize, res)
        elif shape == 'cube':
            self.nb_shape = cube(nbdim, nbsize, res)
        else:
           raise RuntimeError('shape should be \'fast_cube\' or \'sphere\' or \'cube\'.')

    def compute_offsets(self):

    #    Compute neighbor offsets index.

        offsets = self.nb_shape.compute_offsets()

        # flatten data and get nonzero voxel index
        maskidx = np.arange(self.voxnum)
        # iteratly compute neighbor index for each voxel
        volnb = []
        dims = self.data.shape
        imgidx3d = np.array(np.unravel_index(self.imgidx1d, dims))
        imgidx3d = imgidx3d.T[0]
        for v in range(self.voxnum):
            v3d = imgidx3d[v,:]
            nb3d = v3d + offsets.T

            imgnb = is_in_image(nb3d, dims)

            nb1d = np.ravel_multi_index(nb3d[imgnb, :].T, dims)
            masnb = (self.data1d[nb1d] > 0)
            v_offsets = offsets.T[imgnb]
            v_offsets = v_offsets[masnb]
            volnb.append([self.imgidx1d[v], nb1d[masnb]])

        return volnb
