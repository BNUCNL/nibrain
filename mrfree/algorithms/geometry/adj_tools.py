"""
Provide tools for get or make matrix, faces, or other forms that reflect adjacent relationships of brain surface.
"""
import numpy as np
from mrfree.core.region import Region


def mk_adjmatrix(region, mask=None):
    """
    Get adjacency matrix of region, and apply mask if it is given.

    Parameters
    ----------
        region: an instance of Region class
        mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Returns
    -------
        adjmatrix: adjacency matrix of (subj_id, hemi, surf), if mask=None, then shape = (n_vertexes, n_vertexes).
    """
    assert isinstance(region, Region), "Input 'region' should be an instance of Region."
    adj = faces_to_adjmatrix(region.geometry.faces, mask)
    return 0.5 * (adj + adj.T)


def faces_to_edges(faces):
    """
    Build edges array from faces.

    Parameters
    ----------
        faces: triangles mesh of brain surface, shape=(n_mesh, 3).

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
    return edges


def edges_to_adjmatrix(edges):
    """
    Build edges array from faces.

    Parameters
    ----------
        edges: edges of brain surface mesh, shape=(n_edges, 2)

    Returns
    -------
        adj_matrix: adj matrix that reflect linkages of edges, shape = (n_vertexes, n_vertexes).
    """
    vertexes = np.unique(edges)
    n_vertexes = len(vertexes)
    adj_matrix = np.zeros((n_vertexes, n_vertexes))
    for edge in edges:
        adj_matrix[np.where(vertexes == edge[0]), np.where(vertexes == edge[1])] = 1
    adj_matrix[np.where((adj_matrix + adj_matrix.T) > 0)] = 1
    return adj_matrix


def faces_to_adjmatrix(faces, mask=None):
    """
    Build adjacency matrix by faces.

    Parameters
    ----------
        faces: triangles mesh of brain surface, shape=(n_mesh, 3).
        mask: binary array, 1 for region of interest and 0 for others, shape = (n_vertexes,).

    Returns
    -------
        adj_matrix: adj matrix that reflect linkages of faces, shape = (n_vertexes, n_vertexes).
    """
    adj = edges_to_adjmatrix(faces_to_edges(faces))
    if mask:
        adj = np.delete(adj, mask, axis=0)
        adj = np.delete(adj, mask, axis=1)
    return adj


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


def mk_label_adjfaces(label_image, faces):
    """
    Calculate faces of labels in label_image, based on faces of vertexes.

    Parameters
    ----------
        label_image: labels of vertexes, shape = (n, ).
        faces: faces of vertexes, its shape depends on surface, shape = (m, 3).

    Returns
    -------
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


def get_verts_faces(verts, faces):
    """
    Get faces of verts based on faces of all vertexes.

    Parameters
    ----------
        verts: a set of vertices, shape = (k,)
        faces: faces of vertexes, its shape depends on surface, shape = (n_faces, 3).

    Returns
    -------
        faces of verts, shape = (m, 3)
    """
    verts_faces = np.empty((0, 3))
    for vert in verts:
        verts_faces = np.append(verts_faces, faces[np.where(faces == vert)[0]], axis=0)
    verts_faces_rde = np.array(list(set([tuple(column) for column in verts_faces])))  # remove duplicate elements
    return verts_faces_rde


def get_verts_edges(verts, edges):
    """
    Get edges of verts based on edges of all vertexes.

    Parameters
    ----------
        verts: a set of vertices, shape = (k,)
        edges: edges of brain surface mesh, shape=(n_edges, 2)

    Returns
    -------
        edges of verts, shape = (m, 2)
    """
    verts_edges = np.empty((0, 2))
    for vert in verts:
        verts_edges = np.append(verts_edges, edges[np.where(edges == vert)[0]], axis=0)
    verts_edges_rde = np.array(list(set([tuple(column) for column in verts_edges])))  # remove duplicate elements
    return verts_edges_rde


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
