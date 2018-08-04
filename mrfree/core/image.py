#!/usr/bin/env python
# attributes class

class Image(object):
    def __init__(self, data, space, itype, ras2vox, voxsize, dims, src=None):
        """
              Init image

              Parameters
              ----------
              data: image data, a 3d or 4d array.
              itype: image type, a string.
              ras2vox: transform matrix from ras coords to voxel coords, a 4x4 array
              voxsize: voxel size, a 3x1 array
              dims: image dimensions, a 3x1 or 4x1 array
              src: source of the image data, a string.
              """

        self.data = data
        self.space = space
        self.itype = itype
        self.ras2vox = ras2vox
        self.voxsize = voxsize
        self.dims = dims
        self.src = src


