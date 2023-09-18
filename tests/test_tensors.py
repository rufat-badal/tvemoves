from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D
import numpy as np
from pytest import approx


def test_vector():
    n = 1000

    v_numpy = np.random.rand(n)
    v_data = list(v_numpy)
    v = Vector(v_data)
    w_numpy = np.random.rand(n)
    w_data = list(w_numpy)
    w = Vector(w_data)
    s = np.random.rand()

    assert v._data == v_data
    assert w._data == w_data
    assert np.array((v + w)._data) == approx(v_numpy + w_numpy)
    assert np.array((v - w)._data) == approx(v_numpy - w_numpy)
    assert np.array((-v)._data) == approx(-v_numpy)
    assert all(v[i] == v_numpy[i] for i in range(v.shape[0]))
    assert np.array((s * v)._data) == approx(s * v_numpy)
    assert np.array((w * s)._data) == approx(w_numpy * s)
    assert v.dot(w) == approx(v_numpy.dot(w_numpy))
    assert v.normsqr() == approx(v.dot(v))


def test_matrix():
    A_shape = (100, 200)
    B_shape = A_shape
    C_shape = (200, 50)
    AC_shape = (A_shape[0], C_shape[1])
    D_shape = (7, 7)

    A_numpy = np.random.rand(*A_shape)
    A_data = list(A_numpy)
    A = Matrix(A_data)
    B_numpy = np.random.rand(*B_shape)
    B_data = list(B_numpy)
    B = Matrix(B_data)
    C_numpy = np.random.rand(*C_shape)
    C_data = list(C_numpy)
    C = Matrix(C_data)
    D_numpy = np.random.rand(*D_shape)
    D_data = list(D_numpy)
    D = Matrix(D_data)
    s = np.random.rand()

    assert A._data == A_data
    assert A.shape == A_shape
    assert B._data == B_data
    assert B.shape == B_shape
    assert C._data == C_data
    assert C.shape == C_shape
    assert D._data == D_data
    assert D.shape == D_shape
    assert np.array((A + B)._data) == approx(A_numpy + B_numpy)
    assert np.array((A - B)._data) == approx(A_numpy - B_numpy)
    assert np.array((-A)._data) == approx(-A_numpy)
    assert all(
        A[i, j] == A_numpy[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])
    )
    assert (A @ C).shape == AC_shape
    assert np.array((A @ C)._data) == approx(A_numpy @ C_numpy)
    assert np.array((A.T)._data) == approx(np.transpose(A_numpy))
    assert A.trace() == approx(np.trace(A_numpy))
    assert D.det() == approx(np.linalg.det(D_numpy))
    assert A.dot(B) == approx(A_numpy.ravel().dot(B_numpy.ravel()))
    assert A.normsqr() == approx(A.dot(A))
    assert np.array((s * A)._data) == approx(s * A_numpy)
    assert np.array((C * s)._data) == approx(C_numpy * s)


def test_tensor3d():
    T_shape = (100, 200, 50)
    T_numpy = np.random.rand(*T_shape)
    T_data = list(T_numpy)
    T = Tensor3D(T_data)

    assert T._data == T_data
    assert T.shape == T_shape
    assert T.normsqr() == approx(T_numpy.ravel().dot(T_numpy.ravel()))
