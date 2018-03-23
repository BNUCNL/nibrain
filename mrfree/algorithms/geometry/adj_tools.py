"""
Provide tools for get or make matrix, faces, or other forms that reflect adjacent relationships of brain surface.
"""
import numpy as np
from mrfree.core.region import Region


def get_Region_adjmatrix(region, mask=None):
    """
    Get adjacency matrix of region, and apply mask if it is given.

    Parameters
    ----------
        regionA: an instance of Region class
        mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Returns
    -------
        adjmatrix: adjacency matrix of (subj_id, hemi, surf), if mask=None, then shape = (n_vertexes, n_vertexes).
    """
    assert isinstance(region, Region), "Input 'region' should be an instance of Region."
    adj = faces_to_adjmatrix(region.geometry.faces, mask)
    return 0.5 * (adj + adj.T)


def nonconnected_labels(labels, faces):
    """
    Check if every label in labels is a connected component.

    Parameters
    ----------
        labels: cluster labels, shape = [n_samples].
        faces: contain triangles of brain surface.

    Returns
    -------
        label list of nonconnected labels, if None, return [].

    Notes
    -----
        1. the max label number in labels should be assigned to the medial wall.
        2. data with the max label number will be omitted.
    """
    max_label = np.max(labels)
    label_list = []
    for i in range(max_label):
        vertexes = np.array(np.where(labels == i)).flatten()
        visited = []
        neighbors = [vertexes[0]]

        while neighbors:
            vertex = neighbors.pop(0)
            visited.append(vertex)
            neigh = np.unique(faces[np.where(faces == vertex)[0]])
            for vert in neigh:
                if vert in vertexes:
                    if (vert not in visited) and (vert not in neighbors):
                            neighbors.append(vert)

        for vert in vertexes:
            if vert not in visited:
                print("Label %i is not a connected component." % i)
                label_list.append(i)
                break
    return label_list


def connected_conponents_labeling(vertexes, faces):
    """
    Finding connected_conponents of vertexes according to its faces.

    Parameters
    ----------
        vertexes: a set of vertexes that contain several connected component.
        faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).

    Return
    ------
        marks: marks of vertexes, used to split vertexes into different connected components.
    """
    mark = 0
    marks = np.zeros_like(vertexes)

    for vertex in vertexes:
        if marks[np.where(vertexes == vertex)[0]] != 0:
            continue

        mark = mark + 1
        neighbors = [vertex]
        while neighbors:
            vert = neighbors.pop(0)
            marks[np.where(vertexes == vert)[0]] = mark
            neigh = np.unique(faces[np.where(faces == vert)[0]])
            for vert in neigh:
                if vert in vertexes:
                    if (marks[np.where(vertexes == vert)[0]] == 0) and (vert not in neighbors):
                        neighbors.append(vert)
                        marks[np.where(vertexes == vert)[0]] = mark
        if np.all(marks):
            break
    return marks


def merge_small_parts(data, labels, faces, parcel_size):
    """
    Merge small nonconnected parts of labels to its most correlated neighbor.

    If the parcel size of a connected component in a nonconnected label is smaller thatn `parcel_size`, then this
      component will be merged (modify its label) into its neighbors according to the correlation of `data` between
      these parcels.

    Parameters
    ----------
        data: time series that used to check correlation, shape = (n_vertexes, n_features).
        labels: labeling of all vertexes, shape = (n_vertexes, ).
        faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).
        parcel_size: vertex number in a connected component used as threshold, if size of a parcel smaller than
                     parcel_size, then this parcel will be merged.

    Return
    ------
        result_label: labels after merging small parcel.
    """
    nonc_labels = nonconnected_labels(labels, faces)
    result_label = np.copy(labels)
    for nonc_label in nonc_labels:
        vertexes = np.where(labels == nonc_label)[0]
        marks = connected_conponents_labeling(vertexes, faces)

        for m in np.unique(marks):
            verts = vertexes[np.where(marks == m)]
            print("small cluster: {0}: {1}: {2}".format(nonc_label, m, verts.shape))

            if verts.shape[0] < parcel_size:
                verts_data = np.mean(data[verts], axis=0)
                verts_neigh_faces = get_verts_faces(verts, faces)
                neigh_labels = np.setdiff1d(np.unique(result_label[np.unique(verts_neigh_faces).astype(int)]),
                                            nonc_label)
                temp_corr = None

                for neigh_label in neigh_labels:
                    neigh_data = np.mean(data[np.where(result_label == neigh_label)], axis=0)
                    neigh_corr = np.corrcoef(neigh_data, verts_data)[0][1]
                    if neigh_corr > temp_corr:
                        temp_corr = neigh_corr
                        labelid = neigh_label
                print("Set label {0} to verts, correlation: {1}.".format(labelid, temp_corr))
                result_label[verts] = labelid
    return result_label


def split_connected_components(labels, faces):
    """
    Split connected components in same label into defferent labels.

    Parameters
    ----------
        labels: labeling of all vertexes, shape = (n_vertexes, ).
        faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).

    Return
    ------
        result_label: labels after spliting connected components in same label.
    """
    nonc_labels = nonconnected_labels(labels, faces)
    new_label = np.max(labels) + 1
    result_label = np.copy(labels)
    for nonc_label in nonc_labels:
        vertexes = np.where(labels == nonc_label)[0]
        marks = connected_conponents_labeling(vertexes, faces)

        for m in np.unique(marks):
            verts = vertexes[np.where(marks == m)]
            print("small cluster: {0}: {1}: {2}".format(nonc_label, m, verts.shape))
            if m > 1:  # keep label of group m==1.
                result_label[verts] = new_label
                new_label = new_label + 1
    print("Label number: {0}".format(np.max(result_label)))
    return result_label
