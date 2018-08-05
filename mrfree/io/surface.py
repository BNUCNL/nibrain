
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
