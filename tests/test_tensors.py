from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D
import numpy as np
from pytest import approx


def test_vector():
    n = 10
    v_numpy = np.random.rand(n)
    v_data = list(v_numpy)
    w_numpy = np.random.rand(n)
    w_data = list(w_numpy)
    v = Vector(v_data)
    assert v._data == v_data
    w = Vector(w_data)
    assert w._data == w_data
    s = np.random.rand()

    assert np.array((v + w)._data) == approx(v_numpy + w_numpy)
    assert np.array((v - w)._data) == approx(v_numpy - w_numpy)
    assert np.array((-v)._data) == approx(-v_numpy)
    assert np.array((s * v)._data) == approx(s * v_numpy)
    assert np.array((w * s)._data) == approx(w_numpy * s)
    assert all(v[i] == v_numpy[i] for i in range(v.shape[0]))
    assert v.dot(w) == approx(v_numpy.dot(w_numpy))
    assert v.normsqr() == approx(v.dot(v))
