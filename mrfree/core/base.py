import numpy as np

def intersect2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    ia = find2d(A, C)
    ib = find2d(B, C)
    return C,ia, ib


def exclude2d(A, B):
    C = intersect2d(A, B)
    ia = find2d(A, C)
    A = np.delete(A, ia, axis=0)
    return A,ia


def find2d(A, B):
    nrows, ncols = B.shape
    ia = []
    for i in range(nrows):
        ia.append(np.where(A == B[i, :]).all(axis=1).tolist())
    return ia


if __name__ == "__main__":
    # test intersect2d
    A = np.array([[1, 4], [2, 5], [3, 6]])
    B = np.array([[1, 4], [3, 6], [7, 8]])
    C = intersect2d(A, B)


    # test exclude2d
    A = np.array([[1, 4], [2, 5], [3, 6]])
    B = np.array([[1, 4], [3, 6], [7, 8]])
    C = exclude2d(A, B)






