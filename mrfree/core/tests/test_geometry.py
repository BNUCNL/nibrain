
from mrfree.core.geometry import (Points,Lines)
import numpy as np
# import pytest


def test_points():
    coords = np.random.rand(10,3)
    id = np.arange(len(coords))
    P = Points(coords, id)
    print P.coords, P.id

    P.coords = np.random.rand(5,3)
    P.id = np.arange(len(P.coords))
    print P.coords, P.id


def test_lines():
    coords = [np.array([[0, 0., 0.9],
                  [1.9, 0., 0.]]),
        np.array([[0.1, 0., 0],
                  [0, 1., 1.],
                  [0, 2., 2.]]),
        np.array([[2, 2, 2],
                  [3, 3, 3]])]
    id = np.arange(len(coords))
    L = Lines(coords, id)
    print L.coords, L.id


    L.coords = np.delete(L.coords,1)
    L.id = np.delete(L.id,1)
    print L.coords, L.id


if __name__ == "__main__":
    test_points()
    test_lines()