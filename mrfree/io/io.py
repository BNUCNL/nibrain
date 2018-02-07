# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from nibabel import freesurfer
from nibabel.spatialimages import ImageFileError


def load_surf_geom(surf_file, label=None):
    """
    Load surface geometry.

    Parameters
    ----------
    surf_file : Surface file path
            specified as a filename (single file).
    label : Array or None
            a set of id of vertices, shape=(k,).

    Return
    ------
    Surface geometry, faces, label

    """
    if not os.path.exists(surf_file):
        print 'surf file does not exist!'
        return None

    if surf_file.endswith(('.inflated', '.white', '.pial', '.orig')):
        coords, faces = freesurfer.read_geometry(surf_file)
    elif surf_file.endswith('.surf.gii'):
        geo_img = nib.load(surf_file)
        coords = geo_img.darray[0].data
        faces = geo_img.darray[1].data
    else:
        suffix = os.path.split(surf_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if label is not None:
        if label.max() < len(coords):
            coords = coords[label]
            verts_faces = np.empty((0, 3))
            for v in label:
                faces = np.append(verts_faces, faces[np.where(faces == v)[0]], axis=0)
            faces = np.array(list(set([tuple(c) for c in faces])))
            faces_v = np.array(list(set(faces.flatten())))
            remove_faces_index = np.empty((0,)).astype(int)
            for vv in faces_v:
                if vv not in label:
                    remove_faces_index = np.append(remove_faces_index, np.where(faces == vv)[0])
            faces_f = np.empty((0, 3))
            for f in range(len(faces)):
                if f not in remove_faces_index:
                    faces_f = np.append(faces_f, faces[f], axis=0)

            return coords, faces_f, label
        else:
            raise ValueError("Data dimension does not match.")
    else:
        return coords, faces, label


def load_vol_geom(vol_file, mask=None):
    """
    Load volume geometry.

    Parameters
    ----------
    vol_file : Volume file path
            Nifti dataset, specified as a filename (single file).
    mask : Array or None
            a binary(label) volume.

    Return
    ------
    Volume data, xform

    """
    if not os.path.exists(vol_file):
        print 'vol file does not exist!'
        return None

    if (vol_file.endswith('.nii.gz')) | (vol_file.endswith('.nii') & vol_file.count('.') == 1):
        img = nib.load(vol_file)
        coords = img.get_data()
        xform = img.affine
    else:
        suffix = os.path.split(vol_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if mask is not None:
        if coords.shape == mask.shape:
            i, j, k = np.where(mask != 0)
            coords_ijk = zip(i, j, k)
            coords_ijk = np.append(coords_ijk, 1)
            mni = np.dot(img.affine, coords_ijk)
            return mni, xform
        else:
            raise ValueError("Data dimension does not match.")

    else:
        return coords, xform


def load_surf_scalar(surf_scalar_file, label=None):
    """
    Load surface scalar.

    Parameters
    ----------
    surf_scalar_file : Surface scalar file path
            specified as a filename (single file).
    label : Array or None
            a set of id of vertices, shape=(k,).

    Return
    ------
    Surface scalar data

    """
    if not os.path.exists(surf_scalar_file):
        print 'surf scalar file does not exist!'
        return None

    if surf_scalar_file.endswith(('.curv', '.sulc', '.volume', '.thickness', '.area')):
        data = np.expand_dims(freesurfer.read_morph_data(surf_scalar_file), axis=-1)
    elif surf_scalar_file.endswith(('.shape.gii', '.func.gii')):
        data = np.expand_dims(nib.load(surf_scalar_file).darrays[0].data, axis=-1)
    elif surf_scalar_file.endswith(('.mgz', '.mgh')):
        data = nib.load(surf_scalar_file).get_data()
        data = data.reshape((data.shape[0], data.shape[-1]))
    elif surf_scalar_file.endswith(('.dscalar.nii', '.dseries.nii')):
        data = nib.load(surf_scalar_file).get_data()
        data = data.T
    elif surf_scalar_file.endswith('.label.gii'):
        data = np.expand_dims(nib.load(surf_scalar_file).darrays[0].data, axis=-1)
    elif surf_scalar_file.endswith('.dlabel.nii'):
        data = nib.load(surf_scalar_file).get_data().T
    elif surf_scalar_file.endswith('.label'):
        data = np.expand_dims(freesurfer.read_label(surf_scalar_file), axis=-1)
    elif surf_scalar_file.endswith('.annot'):
        data, _, _ = freesurfer.read_annot(surf_scalar_file)
    else:
        suffix = os.path.split(surf_scalar_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if label is not None:
        if len(label) == len(data):
            data = data[label]
            return data
        else:
            raise ValueError("Data dimension does not match.")
    else:
        return data


def load_vol_scalar(vol_scalar_file, mask=None):
    """
    Load volume scalar.

    Parameters
    ----------
    vol_scalar_file : Volume scalar file path
            Nifti dataset, specified as a filename (single file).
    mask : Array or None
            a binary(label) volume.

    Return
    ------
    Volume scalar data

    """
    if not os.path.exists(vol_scalar_file):
        print 'vol scalar file does not exist!'
        return None

    if (vol_scalar_file.endswith('.nii.gz')) | (vol_scalar_file.endswith('.nii') & vol_scalar_file.count('.') == 1):
        img = nib.load(vol_scalar_file)
        data = img.get_data()
    elif vol_scalar_file.endswith(('.mgz', '.mgh')):
        data = nib.load(vol_scalar_file).get_data()
        data = data.reshape((data.shape[0], data.shape[-1]))
    elif vol_scalar_file.endswith(('.dscalar.nii', '.dseries.nii')):
        data = nib.load(vol_scalar_file).get_data()
        data = data.T
    elif vol_scalar_file.endswith('.dlabel.nii'):
        data = nib.load(vol_scalar_file).get_data().T
    else:
        suffix = os.path.split(vol_scalar_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if mask is not None:
        if data.shape == mask.shape:
            i, j, k = np.where(mask != 0)
            coords_ijk = zip(i, j, k)
            coords_ijk = np.append(coords_ijk, 1)
            mni = np.dot(img.affine, coords_ijk)
            return mni
        else:
            raise ValueError("Data dimension does not match.")

    else:
        return data
