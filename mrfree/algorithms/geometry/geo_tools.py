#!/usr/bin/env python

"""
Provide tools for get or make matrix, faces, or other forms that reflect adjacent relationships of brain surface.
"""
from itertools import combinations

import numpy as np


def faces_to_edges(faces, mask=None):
    """
    Build edges array from faces.

    Parameters
    ----------
    faces: triangles mesh of brain surface, shape=(n_mesh, 3).
    mask: binary array, 1 for region of interest and 0 for others, shape=(n_vertexes,).

    Returns
    -------
    edges: array, edges of brain surface mesh, shape=(n_edges, 2)
    """
    from itertools import combinations

    edges = np.empty((0, 2))
    for face in faces:
        for edge in combinations(face, 2):
            if np.any(np.all(edge == edges, axis=1)):  # check whether edge in edges
                continue
            if np.any(np.all(edge[::-1] == edges, axis=1)):  # check whether edge in edges
                continue
            edges = np.append(edges, np.reshape(edge, (1, 2)),  axis=0)
    edges = _apply_mask(edges, mask)
    return edges


def edges_to_adjmatrix(edges, mask=None):
    """
    Build edges array from faces.

    Parameters
    ----------
    edges: edges of brain surface mesh, shape=(n_edges, 2).
    mask: binary array, 1 for region of interest and 0 for others, shape=(n_vertexes,).

    Returns
    -------
    adjm: adjacency matrix that reflect linkages of edges, shape = (n_vertexes, n_vertexes).
    """
    vertexes = np.unique(edges)
    n_vertexes = len(vertexes)
    adjm = np.zeros((n_vertexes, n_vertexes))

    for edge in edges:
        adjm[edge[0], edge[1]] = 1
        adjm[edge[1], edge[0]] = 1
    adjm = _apply_mask_on_adjm(adjm, mask=mask)
    return adjm


def rings_to_adjmatrix(ring):
    """
    Generate adjacent matrix from ringlist
    
    Parameters:
    ----------
    ring: list of ring node, the format of ring list like below
          [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
          each element correspond to a index (index means a vertex)
     
    Return:
    ----------
    adjmatrix: adjacent matrix 
    """
    assert isinstance(ring, list), "ring should be a list"
    node_number = len(ring)
    adjmatrix = np.zeros((node_number, node_number))
    for i,e in enumerate(ring):
        for j in e:
             adjmatrix[i,j] = 1
    return adjmatrix


def faces_to_adjmatrix(faces, mask=None):
    """
    Build adjacency matrix by faces.

    Parameters
    ----------
    faces: triangles mesh of brain surface, shape=(n_mesh, 3).
    mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Returns
    -------
    adjm: adjacency matrix that reflect linkages of faces, shape = (n_vertexes, n_vertexes).
    """
    vertexes = np.unique(faces)
    n_vertexes = len(vertexes)
    adjm = np.zeros((n_vertexes, n_vertexes))

    for face in faces:
        for edge in combinations(face, 2):
            adjm[edge[0], edge[1]] = 1
            adjm[edge[1], edge[0]] = 1
    adjm = _apply_mask_on_adjm(adjm, mask=mask)
    return adjm


def mk_label_adjmatrix(label_image, adjmatrix):
    """
    Calculate adjacent matrix of labels in label_image, based on adjacent matrix of vertexes.

    Parameters
    ----------
    label_image: labels of vertexes, shape = (n, ), n is number of vertexes.
    adjmatrix: adjacent matrix of vertexes, shape = (n, n).

    Returns
    -------
    label_adjmatrix: adjacent matrix of labels, shape = (l, l), l is number of labels.

    Notes
    -----
    1. for large number of vertexes, this method may cause memory error, try to use mk_label_adjfaces().
    """
    labels = np.unique(label_image)
    l, n = len(labels), len(label_image)
    temp_matrix = np.zeros((l, n))
    label_adjmatrix = np.zeros((l, l))
    for i, label in enumerate(labels):
        temp_matrix[i, :] = np.sum(adjmatrix[np.where(label_image == label)[0], :], axis=0)

    for i, label in enumerate(labels):
        label_adjmatrix[:, i] = np.sum(temp_matrix[:, np.where(label_image == label)[0]], axis=1).T

    # making binary adjmatrix
    label_adjmatrix[np.where(label_adjmatrix > 0)] = 1
    label_adjmatrix[range(l), range(l)] = 0
    return label_adjmatrix


def _apply_mask(data, mask=None):
    """
    Apply mask to faces or edges by delete masked data.

    Parameters
    ----------
    data: inout data, should be faces or edges.
    mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Return
    ------
    result: return data if no mask, else return masked data, and its shape may change.
    """
    if not mask:
        return data

    mask_1dim = np.reshape(mask, (-1))
    masked_verts = np.where(mask_1dim == 0)[0]
    index = []
    for vert in masked_verts:
        index = np.concatenate([index, np.where(data == vert)[0]])
    index = np.unique(index).astype(np.int)
    result = np.delete(data, index, axis=0)
    return result


def _apply_mask_on_adjm(adjm, mask=None):
    """
    Apply mask to adjacency matrix by delete masked data.

    Parameters
    ----------
    adjm: input data, should be faces or edges.
    mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Return
    ------
    adjm: return data if no mask, else return masked adjmatrix, and its shape may change.
    """
    if not mask:
        return adjm

    adjm = np.delete(adjm, mask, axis=0)
    adjm = np.delete(adjm, mask, axis=1)
    return adjm


def mk_label_adjfaces(label_image, faces):
    """
    Calculate faces of labels in label_image, based on faces of vertexes.

    Parameters
    ----------
    label_image: labels of vertexes, shape = (n, ).
    faces: faces of vertexes, its shape depends on surface, shape = (m, 3).

    Return
    ------
    label_faces: faces of labels, shape = (l, 3).
    """
    label_face = np.copy(faces)
    for i in faces:
        label_face[np.where(faces == i)[0]] = [label_image[i[0]], label_image[i[1]], label_image[i[2]]]
    label_faces_rde = np.array(list(set([tuple(column) for column in label_face])))  # remove duplicate elements
    label_faces = np.empty((0, 3))
    for column in label_faces_rde:
        if np.unique(column).shape[0] != 1:
            label_faces = np.append(label_faces, column, axis=0)  # keep face elements only
    return np.array(label_faces)


def get_verts_faces(vertices, faces, keep_neighbor=False):
    """
    Get faces of verts based on faces of all vertexes.

    Parameters
    ----------
    vertices: a set of vertices, shape = (k,)
    faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).
    keep_neighbor: whether to keep 1-ring neighbor of verts in the result or not,
        default is False.

    Return
    ------
    verts_faces_rde: faces of verts, shape = (m, 3)
    """
    verts_faces = np.empty((0, 3))
    for vert in vertices:
        verts_faces = np.append(verts_faces, faces[np.where(faces == vert)[0]], axis=0)
    # remove duplicate elements
    verts_faces_rde = np.array(list(set([tuple(column) for column in verts_faces])))  # remove duplicate elements

    if not keep_neighbor:
        verts_all = np.unique(verts_faces_rde)
        for vert in verts_all:
            if vert not in vertices:
                verts_faces_rde = np.delete(verts_faces_rde,
                                            np.where(verts_faces_rde == vert)[0],
                                            axis=0)
    return verts_faces_rde


def get_verts_edges(vertices, edges, keep_neighbor=False):
    """
    Get edges of verts based on edges of all vertexes.

    Parameters
    ----------
    vertices: a set of vertices, shape = (k,)
    edges: edges of brain surface mesh, shape=(n_edges, 2)
    keep_neighbor: whether to keep 1-ring neighbor of vertices in the result or not,
        default is False.

    Return
    ------
    verts_edges_rde: edges of verts, shape = (m, 2)
    """
    verts_edges = np.empty((0, 2))
    for vert in vertices:
        verts_edges = np.append(verts_edges, edges[np.where(edges == vert)[0]], axis=0)
    # remove duplicate elements
    verts_edges_rde = np.array(list(set([tuple(column) for column in verts_edges])))

    if not keep_neighbor:
        verts_all = np.unique(verts_edges_rde)
        for vert in verts_all:
            if vert not in vertices:
                verts_edges_rde = np.delete(verts_edges_rde,
                                            np.where(verts_edges_rde == vert)[0],
                                            axis=0)
    return verts_edges_rde


def nonconnected_labels(labels, faces, showinfo=False):
    """
    Check if every label in labels is a connected component.

    Parameters
    ----------
    labels: cluster labels, shape = [n_samples].
    faces: contain triangles of brain surface.
    showinfo: whether print details or not, default is False.

    Return
    ------
    label list of nonconnected labels, if not found, return [].

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
                if showinfo:
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


def split_connected_components(labels, faces, showinfo=False):
    """
    Split connected components in same label into defferent labels.

    Parameters
    ----------
    labels: labeling of all vertexes, shape = (n_vertexes, ).
    faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).
    showinfo: whether print details or not, default is False.

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
            if showinfo:
                print("small cluster: {0}: {1}: {2}".format(nonc_label, m, verts.shape))
            if m > 1:  # keep label of group m==1.
                result_label[verts] = new_label
                new_label = new_label + 1
    print("Label number after processing: {0}".format(np.max(result_label)))
    return result_label


def merge_small_parts(data, labels, faces, parcel_size, showinfo=False):
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
    showinfo: whether print details or not, default is False.

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
            if showinfo:
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
                if showinfo:
                    print("Set label {0} to verts, correlation: {1}.".format(labelid, temp_corr))
                result_label[verts] = labelid
    return result_label
