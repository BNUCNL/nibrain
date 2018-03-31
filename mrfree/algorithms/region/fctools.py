"""
Calculate different types of functional correlation that maybe useful in natural stimulus analysis.
Input file format: *.mgh (output format: *.mgh)
the input file(*.mgh) is converted from *.nii.gz file, in order to display by pysurfer.
isfc: Inter-subject functional correlation.
isc: Inter-subject correlation.
fc: Functional correlation.
"""
import numpy as np
from scipy.spatial.distance import cdist


def isfc(data1, data2):
    """
    Cal functional connectivity between data1 and data2.

    Parameters
    ----------
        data1: used to calculate functional connectivity, shape = [n_samples1, n_features].
        data2: used to calculate functional connectivity, shape = [n_samples2, n_features].

    Returns
    -------
        isfc: functional connectivity map of data1 and data2, shape = [n_samples1, n_samples2].

    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. n_features should be the same in data1 and data2.
    """
    return 1 - cdist(data1, data2, metric='correlation')


def isc(data1, data2):
    """
    Cal ISC between data1 and data2 vertex by vertex.

    Parameters
    ----------
        data1: used to calculate functional connectivity, shape = [n_samples, n_features].
        data2: used to calculate functional connectivity, shape = [n_samples, n_features].

    Returns
    -------
        isc: point-to-point functional connectivity list of data1 and data2, shape = [n_samples, ].

    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. [n_samples, n_features] should be the same in data1 and data2.
    """
    assert data1.shape == data2.shape, 'data1 and data2 should have the same shape.'
    corr = np.array([np.corrcoef(data1[i], data2[i])[0, 1] for i in range(data1.shape[0])])
    return corr


def wsfc(data):
    """
    Cal within subject functional connectivity of data.

    Parameters
    ----------
        data: used to calculate functional connectivity, shape = [n_samples, n_features].

    Returns
    -------
        wsfc: functional connectivity map of data, shape = [n_samples, n_samples].

    Notes
    -----
        1. data should be 2-dimensional.
    """
    return isfc(data, data)
