"""Test tensor classes."""

import numpy as np
from pytest import approx
from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D


def test_vector() -> None:
    """Test vector class."""
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

    assert v.data == v_data
    assert w.data == w_data
    assert (v + w).data == approx(v_numpy + w_numpy)
    assert (v - w).data == approx(v_numpy - w_numpy)
    assert (-v).data == approx(-v_numpy)
    assert (v.numpy() == v_numpy).all()
    assert v.dot(w) == approx(v_numpy.dot(w_numpy))
    assert v.normsqr() == approx(v.dot(v))
    assert v.norm() == approx(np.sqrt(v.normsqr()))
    assert (s * v).data == approx(s * v_numpy)
    assert (w * s).data == approx(w_numpy * s)
    assert (v / d).data == approx(v_numpy / d)
    assert v.reshape(m, k).data == approx(v_numpy.reshape(m, k))
    assert v.stack(w).data == approx(np.stack((v_numpy, w_numpy)))
    assert v.map(lambda x: -x).data == (-v).data

    def square(x):
        return x**2

    square_vectorized = np.vectorize(square)
    assert v.map(square).data == approx(square_vectorized(v_numpy))


def test_matrix() -> None:
    """Test matrix class."""
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

    assert A.data == A_data
    assert A.shape == A_shape
    assert B.data == B_data
    assert B.shape == B_shape
    assert C.data == C_data
    assert C.shape == C_shape
    assert D.data == D_data
    assert D.shape == D_shape
    assert (A + B).data == approx(A_numpy + B_numpy)
    assert (A - B).data == approx(A_numpy - B_numpy)
    assert (-A).data == approx(-A_numpy)
    assert (A.numpy() == A_numpy).all()
    assert (A @ C).shape == AC_shape
    assert (A @ C).data == approx(A_numpy @ C_numpy)
    assert A.transpose().data == approx(np.transpose(A_numpy))
    assert A.trace() == approx(np.trace(A_numpy))
    assert D.det() == approx(np.linalg.det(D_numpy))
    assert A.scalar_product(B) == approx(A_numpy.ravel().dot(B_numpy.ravel()))
    assert A.normsqr() == approx(A.scalar_product(A))
    assert (s * A).data == approx(s * A_numpy)
    assert (C * s).data == approx(C_numpy * s)
    assert (C / d).data == approx(C_numpy / d)
    assert A.flatten().data == approx(A_numpy.flatten())
    assert A.dot(v).data == approx(A_numpy.dot(v_numpy))
    assert A.map(lambda x: -x).data == (-A).data

    def square(x):
        return x**2

    square_vectorized = np.vectorize(square)
    assert A.map(square).data == approx(square_vectorized(A_numpy))
    assert A.stack(B).data == approx(np.stack((A_numpy, B_numpy)))


def test_tensor3d() -> None:
    """Test 3-d tensor class."""
    T_shape = (50, 100, 25)
    T_numpy = np.random.rand(*T_shape)
    T_data = list(T_numpy)
    T = Tensor3D(T_data)

    S_shape = T_shape
    S_numpy = np.random.rand(*S_shape)
    S_data = list(S_numpy)
    S = Tensor3D(S_data)

    assert T.data == T_data
    assert T.shape == T_shape
    assert T.normsqr() == approx(T_numpy.ravel().dot(T_numpy.ravel()))

    def square(x):
        return x**2

    square_vectorized = np.vectorize(square)
    assert T.map(square).data == approx(square_vectorized(T_numpy))
    assert (T + S).data == approx(T_numpy + S_numpy)
    assert (T - S).data == approx(T_numpy - S_numpy)
    assert (-T).data == approx(-T_numpy)
    assert all(
        T[i, j, k] == T_numpy[i, j, k]
        for i in range(T.shape[0])
        for j in range(T.shape[1])
        for k in range(T.shape[2])
    )
