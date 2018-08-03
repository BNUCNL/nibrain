import numpy as np


def intersect2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)


if __name__ == "__main__":
# Test intersect2d

A = np.array([[1, 4], [2, 5], [3, 6]])
B = np.array([[1, 4], [3, 6], [7, 8]])
C = intersect2d(A, B)
