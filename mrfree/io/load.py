# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from nibabel import freesurfer
from nibabel.spatialimages import ImageFileError


def load_surf_geom(surf_file, surf_label_file=None):
    """
    Load surface geometry.

    Parameters
    ----------
    surf_file : Surface file path
            specified as a filename (single file).
    surf_label_file: Surface label file path
                specified as a filename (single file).

    Return
    ------
    Surface geometry, faces, label

    """

    if surf_file.endswith(('.inflated', '.white', '.pial', '.orig')):
        coords, faces = freesurfer.read_geometry(surf_file)
    elif surf_file.endswith('.surf.gii'):
        geo_img = nib.load(surf_file)
        coords = geo_img.darray[0].data
        faces = geo_img.darray[1].data
    else:
        suffix = os.path.split(surf_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if surf_label_file is not None:
        label = _load_surf_label(surf_label_file)
        if label.max() < len(coords):
            coords = coords[label]
            index = [set(fc).issubset(set(label)) for fc in faces]
            faces = faces[index]

            return coords, faces, label
        else:
            raise ValueError("Data dimension does not match.")
    else:
        return coords, faces, None


def load_vol_geom(vol_file, vol_mask_file=None):
    """
    Load volume geometry.

    Parameters
    ----------
    vol_file : Volume file path
            Nifti dataset, specified as a filename (single file).
    vol_mask_file: Volume mask file path
                Nifti dataset, specified as a filename (single file).

    Return
    ------
    Volume data, xform

    """

    if (vol_file.endswith('.nii.gz')) | (vol_file.endswith('.nii') & vol_file.count('.') == 1):
        img = nib.load(vol_file)
        coords = img.get_data()
        xform = img.affine
    else:
        suffix = os.path.split(vol_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if vol_mask_file is not None:
        mask = _load_vol_mask(vol_mask_file)
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


def load_surf_scalar(surf_scalar_file, surf_label_file=None):
    """
    Load surface scalar.

    Parameters
    ----------
    surf_scalar_file : Surface scalar file path
            specified as a filename (single file).
    surf_label_file: Surface label file path
                specified as a filename (single file).

    Return
    ------
    Surface scalar data

    """

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
        data = data.T
    else:
        suffix = os.path.split(surf_scalar_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    if surf_label_file is not None:
        label = _load_surf_label(surf_label_file)
        if label.max() < len(data):
            data = data[label]
            return data
        else:
            raise ValueError("Data dimension does not match.")
    else:
        return data


def load_vol_scalar(vol_scalar_file, vol_mask_file=None):
    """
    Load volume scalar.

    Parameters
    ----------
    vol_scalar_file : Volume scalar file path
            Nifti dataset, specified as a filename (single file).
    vol_mask_file: Volume mask file path
                Nifti dataset, specified as a filename (single file).

    Return
    ------
    Volume scalar data

    """

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

    if vol_mask_file is not None:
        mask = _load_vol_mask(vol_mask_file)
        if data.shape == mask.shape:
            data = data[mask > 0]
            return data
        else:
            raise ValueError("Data dimension does not match.")

    else:
        return data


def _load_surf_label(surf_label_file):
    """
    Load label or mask of surface.

    Parameters
    ----------
    surf_label_file: Surface label file path
                specified as a filename (single file).

    Return
    ------
    label or mask of surface.
    """
    if surf_label_file.endswith('.label.gii'):
        data = np.expand_dims(nib.load(surf_label_file).darrays[0].data, axis=-1)
    elif surf_label_file.endswith('.dlabel.nii'):
        data = nib.load(surf_label_file).get_data().T
    elif surf_label_file.endswith('.label'):
        data = np.expand_dims(freesurfer.read_label(surf_label_file), axis=-1)
    elif surf_label_file.endswith('.annot'):
        data, _, _ = freesurfer.read_annot(surf_label_file)
        data = data.T
    else:
        suffix = os.path.split(surf_label_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    return data


def _load_vol_mask(vol_mask_file):
    """
    Load label or mask of volume.

    Parameters
    ----------
    vol_mask_file: Volume mask file path
                Nifti dataset, specified as a filename (single file).

    Return
    ------
    label or mask of volume.
    """
    if (vol_mask_file.endswith('.nii.gz')) | (vol_mask_file.endswith('.nii') & vol_mask_file.count('.') == 1):
        img = nib.load(vol_mask_file)
        data = img.get_data()
    else:
        suffix = os.path.split(vol_mask_file)[1].split('.')[-1]
        raise ImageFileError('This file format-{} is not supported at present.'.format(suffix))

    return data
