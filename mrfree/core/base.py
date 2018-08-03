import numpy as np


def intersect2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


def subtract2d(A, B):
    C = intersect2d(A, B)
    nrows, ncols = C.shape
    idx = []
    for i in range(nrows):
        idx.append(np.where(A == C[i,:]).all(axis=1).tolist())
    A = np.delete(A, idx, axis=0)
    return A


if __name__ == "__main__":
    # test intersect2d
    A = np.array([[1, 4], [2, 5], [3, 6]])
    B = np.array([[1, 4], [3, 6], [7, 8]])
    C = intersect2d(A, B)


    # test subtract2d
    A = np.array([[1, 4], [2, 5], [3, 6]])
    B = np.array([[1, 4], [3, 6], [7, 8]])
    C = subtract2d(A, B)






