from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D
import numpy as np
from pytest import approx


def test_vector():
    m = 100
    k = 10
    n = m * k

    v_numpy = np.random.rand(n)
    v_data = list(v_numpy)
    v = Vector(v_data)
    w_numpy = np.random.rand(n)
    w_data = list(w_numpy)
    w = Vector(w_data)
    s = np.random.rand()
    d = np.random.rand() + 1

    assert v._data == v_data
    assert w._data == w_data
    assert (v + w)._data == approx(v_numpy + w_numpy)
    assert (v - w)._data == approx(v_numpy - w_numpy)
    assert (-v)._data == approx(-v_numpy)
    assert all(v[i] == v_numpy[i] for i in range(v.shape[0]))
    assert v.dot(w) == approx(v_numpy.dot(w_numpy))
    assert v.normsqr() == approx(v.dot(v))
    assert (s * v)._data == approx(s * v_numpy)
    assert (w * s)._data == approx(w_numpy * s)
    assert (v / d)._data == approx(v_numpy / d)
    assert v.reshape(m, k)._data == approx(v_numpy.reshape(m, k))
    assert v.vstack(w)._data == approx(np.vstack((v_numpy, w_numpy)))
    assert v.map(lambda x: -x)._data == (-v)._data
    square = lambda x: x**2
    square_vectorized = np.vectorize(square)
    assert v.map(square)._data == approx(square_vectorized(v_numpy))


def test_matrix():
    A_shape = (100, 200)
    B_shape = A_shape
    C_shape = (200, 50)
    AC_shape = (A_shape[0], C_shape[1])
    D_shape = (7, 7)

    A_numpy = np.random.rand(*A_shape)
    A_data = list(A_numpy)
    A = Matrix(A_data)
    v_numpy = np.random.rand(A_shape[1])
    v_data = list(v_numpy)
    v = Vector(v_data)
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
    d = np.random.rand() + 1

    assert A._data == A_data
    assert A.shape == A_shape
    assert B._data == B_data
    assert B.shape == B_shape
    assert C._data == C_data
    assert C.shape == C_shape
    assert D._data == D_data
    assert D.shape == D_shape
    assert (A + B)._data == approx(A_numpy + B_numpy)
    assert (A - B)._data == approx(A_numpy - B_numpy)
    assert (-A)._data == approx(-A_numpy)
    assert all(
        A[i, j] == A_numpy[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])
    )
    assert (A @ C).shape == AC_shape
    assert (A @ C)._data == approx(A_numpy @ C_numpy)
    assert A.transpose()._data == approx(np.transpose(A_numpy))
    assert A.trace() == approx(np.trace(A_numpy))
    assert D.det() == approx(np.linalg.det(D_numpy))
    assert A.scalar_product(B) == approx(A_numpy.ravel().dot(B_numpy.ravel()))
    assert A.normsqr() == approx(A.scalar_product(A))
    assert (s * A)._data == approx(s * A_numpy)
    assert (C * s)._data == approx(C_numpy * s)
    assert (C / d)._data == approx(C_numpy / d)
    assert A.flatten()._data == approx(A_numpy.flatten())
    assert A.dot(v)._data == approx(A_numpy.dot(v_numpy))
    assert A.map(lambda x: -x)._data == (-A)._data
    square = lambda x: x**2
    square_vectorized = np.vectorize(square)
    assert A.map(square)._data == approx(square_vectorized(A_numpy))


def test_tensor3d():
    T_shape = (100, 200, 50)
    T_numpy = np.random.rand(*T_shape)
    T_data = list(T_numpy)
    T = Tensor3D(T_data)

    assert T._data == T_data
    assert T.shape == T_shape
    assert T.normsqr() == approx(T_numpy.ravel().dot(T_numpy.ravel()))
    square = lambda x: x**2
    square_vectorized = np.vectorize(square)
    assert T.map(square)._data == approx(square_vectorized(T_numpy))
