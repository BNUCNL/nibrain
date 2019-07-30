import os
import numpy as np
from mrfree.core.geometry import Mesh
from nibabel import freesurfer
from mrfree.core.surface import Surface


def test_surface():
    surf_mesh_path = '/nfs/e5/stanford/Longitudinal/3Danat/fsaverage/surf/lh.white'
    vertices, faces = freesurfer.read_geometry(surf_mesh_path)

    surf_a = Surface(mesh=Mesh(vertices, faces), space='fsaverage')
    print surf_a.mesh.vertices.shape

    surf_data_path = '/nfs/e5/stanford/Longitudinal/3Danat/fsaverage/surf/lh.curv'
    surf_a.load_data(surf_data_path)
    print surf_a.data.shape

    roi = np.arange(0,50)
    roi_coords = surf_a.get_roi_coords(roi)
    roi_data = surf_a.get_roi_data(roi)
    print roi_coords.shape, roi_data.shape

    surf_mesh_path = '/nfs/e5/stanford/Longitudinal/3Danat/fsaverage/surf/rh.white'
    vertices, faces = freesurfer.read_geometry(surf_mesh_path)

    surf_b = Surface(mesh=Mesh(vertices, faces), space='fsaverage')

    surf_data_path = '/nfs/e5/stanford/Longitudinal/3Danat/fsaverage/surf/rh.curv'
    surf_b.load_data(surf_data_path)
    surf_c = surf_a + surf_b
    print surf_b.mesh.vertices.shape, surf_c.mesh.vertices.shape

if __name__ == "__main__":
    test_surface()
