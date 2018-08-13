
from mrfree.core.geometry import (Points,Lines)
import numpy as np
import pytest


def test_points():
    data = np.random.rand(10,3)
    id = np.arange(len(data))
    src = "Faked points"
    P = Points(data, id, src)
    P.data = np.random.rand(5,3)
    P.id = np.arange(len(P.data))
    P.src = "New faked points"


def test_lines():
    data = [np.array([[0, 0., 0.9],
                  [1.9, 0., 0.]]),
        np.array([[0.1, 0., 0],
                  [0, 1., 1.],
                  [0, 2., 2.]]),
        np.array([[2, 2, 2],
                  [3, 3, 3]])]
    id = np.arange(len(data))
    src = "Faked lines"
    L = Lines(data, id, src)
    L.data =  L.data.remove(1)
    L.id = np.delete(L.id,1)
    L.src = "New faked lines"


