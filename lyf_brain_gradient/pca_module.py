import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from random import shuffle
from os.path import join as pjoin

def concatenate_subject(axis=0,amount=0,fc_dir=None,shuffled=False,scaled=False):
    '''
    concatenate subject fc matrix along an axis.The functional connectivity matrix need to be saved as
    .npy file in a specified directory.The directory should only save fc matrix files.

    Parameters
    ----------
    axis: int, 0 or 1
        The axis that subject matrix will be concatenated along.
    amount: int
        The number of subject matrix that wish to be concatenated.
    fc_dir: str
        The directory of functional connectivity.
    shuffled: bool
        whether the subject list will be shuffled.
    scaled: bool
        whether the matrix will be scaled along the axis that concatenate the subject.
    Returns
    ----------
    result: ndarray
        The concatenated array of fc matrix.

    '''
    subjects = os.listdir(fc_dir)
    if shuffled:
        shuffle(subjects)
    axis0_length = np.load(pjoin(fc_dir,subjects[0])).shape[0]
    axis1_length = np.load(pjoin(fc_dir,subjects[0])).shape[1]
    for number in range(amount):
        if axis == 0:
            concat_axis0_length = amount*axis0_length
            result = np.zeros((concat_axis0_length,axis1_length))
            subject_matrix = np.load(pjoin(fc_dir,subjects[number]))
            result[number*axis0_length:(number+1)*axis0_length, :] = subject_matrix
        elif axis == 1:
            concat_axis1_length = amount*axis1_length
            result = np.zeros((axis0_length,concat_axis1_length))
            subject_matrix = np.load(pjoin(fc_dir,subjects[number]))
            result[:,number*axis1_length:(number+1)*axis1_length] = subject_matrix
    if scaled:
        result = scale(result,axis=axis)
    return result

def calculate_pca(matrix=None,n_components=1,amount=0,isconcatenated=True,axis=0):
    '''
    calculate pca and return its principal components and principal axes.If input matrix is subject-concatenated,
    then return the between-subject average components and axes.

    Parameters
    ----------
    matrix: ndarray
        The input array.
    n_components: int
        The number of components to keep.
    axis: int, 0 or 1
        The axis that subject is concatenated along.
    amount: int
        The number of subject matrix that already concatenated.
    Returns
    ----------
    tuple:include principal components, principal axe, explained variance ratio
    '''
    pca = PCA(n_components = n_components)
    tmp = pca.fit(matrix)
    principal_components = tmp.transform(matrix)
    principal_axe = tmp.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    if isconcatenated:
        if axis == 0:
            tmp_array = np.reshape(principal_components,(amount,n_components,int(np.divide(matrix.shape[0],amount))))
            mean_principal_components = np.divide(np.sum(tmp_array,axis=0),amount)
            return (mean_principal_components,principal_axe,explained_variance_ratio)
        
        if axis == 1:
            tmp_array = np.reshape(principal_axe,(amount,int(np.divide(matrix.shape[1],amount)),n_components))
            mean_principal_axe = np.divide(np.sum(tmp_array,axis=0),amount)
            return (principal_components,mean_principal_axe,explained_variance_ratio)
        
    return (principal_components,principal_axe,explained_variance_ratio)