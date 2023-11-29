"""Test shape function code."""

import sympy as sp
from tvemoves_rufbad.bell_finite_element import (
    _cyclic_permutation,
    _N_first_third,
    _N,
    _N_on_edge,
    _b,
    _c,
    _L,
    _t,
)
from tvemoves_rufbad.tensors import Vector, Matrix
from tvemoves_rufbad.bell_finite_element import (
    shape_function,
    shape_function_jacobian,
    shape_function_hessian_vectorized,
    transform_gradient,
    transform_hessian,
)
from .helpers import random_symbolic_polynomial_2d, random_barycentric_coordinates


def test_cyclic_permutation() -> None:
    """Assure that cyclic permutation restores the original expression."""
    n_first_third_new = _N_first_third
    for _ in range(3):
        n_first_third_new = [
            _cyclic_permutation(ni, _L, _b, _c) for ni in n_first_third_new
        ]
    assert all(ni_new == ni for ni_new, ni in zip(n_first_third_new, _N_first_third))


def test_independence_of_opposite_point() -> None:
    """Check that the shape function on an edge is independent of the point opposite to the edge."""
    for i in range(12, 18):
        assert _N[i].subs(_L[2], 0) == 0

    for i in range(12):
        assert _N[i].subs(_L[2], 0).free_symbols.issubset({_L[0], _L[1], _b[2], _c[2]})

    for ni in _N_on_edge:
        assert ni.free_symbols.issubset({_t, _b[2], _c[2]})


def test_shape_function() -> None:
    """Assure that the shape function can recover a degree 4 polynomial in a triangle.

    see e.g. page 117 in https://people.sc.fsu.edu/~jburkardt/classes/fem_2011/chapter6.pdf
    """

    num_evaluations = 10
    eps = 1e-10

    x, y = sp.symbols("x y")
    triangle_vertices = (Vector([1.0, 2.0]), Vector([5.0, 2.4]), Vector([4.5, 8.2]))
    p1, p2, p3 = triangle_vertices

    poly_symbolic = random_symbolic_polynomial_2d(4, x, y)
    poly_dx_symbolic = sp.diff(poly_symbolic, x)
    poly_dy_symbolic = sp.diff(poly_symbolic, y)
    poly_dxx_symbolic = sp.diff(poly_dx_symbolic, x)
    poly_dxy_symbolic = sp.diff(poly_dx_symbolic, y)
    assert poly_dxy_symbolic == sp.diff(poly_dy_symbolic, x)
    poly_dyy_symbolic = sp.diff(poly_dy_symbolic, y)

    poly = sp.lambdify([x, y], poly_symbolic)
    poly_dx = sp.lambdify([x, y], poly_dx_symbolic)
    poly_dy = sp.lambdify([x, y], poly_dy_symbolic)
    poly_dxx = sp.lambdify([x, y], poly_dxx_symbolic)
    poly_dxy = sp.lambdify([x, y], poly_dxy_symbolic)
    poly_dyy = sp.lambdify([x, y], poly_dyy_symbolic)

    parameters_list = []
    for v in triangle_vertices:
        parameters_list.extend(
            [
                poly(v[0], v[1]),
                poly_dx(v[0], v[1]),
                poly_dy(v[0], v[1]),
                poly_dxx(v[0], v[1]),
                poly_dxy(v[0], v[1]),
                poly_dyy(v[0], v[1]),
            ]
        )
    parameters = Vector(parameters_list)

    def poly_jacobian(x: float, y: float) -> Vector:
        return Vector([poly_dx(x, y), poly_dy(x, y)])

    def poly_hessian(x: float, y: float) -> Matrix:
        return Matrix(
            [[poly_dxx(x, y), poly_dxy(x, y)], [poly_dxy(x, y), poly_dyy(x, y)]]
        )

    for c in random_barycentric_coordinates(num_evaluations):
        c_euclidean = c.u * p1 + c.v * p2 + c.w * p3
        value = poly(*c_euclidean)
        value_approx = shape_function(triangle_vertices, c).dot(parameters)
        assert abs(value - value_approx) < eps

        jacobian = poly_jacobian(*c_euclidean)
        barycentric_jacobian_approx = (
            shape_function_jacobian(triangle_vertices, c).transpose().dot(parameters)
        )
        jacobian_approx = transform_gradient(
            triangle_vertices, barycentric_jacobian_approx
        )
        assert (jacobian - jacobian_approx).norm() < eps

        hessian = poly_hessian(*c_euclidean)
        barycentric_hessian_vectorized_approx = (
            shape_function_hessian_vectorized(triangle_vertices, c)
            .transpose()
            .dot(parameters)
        )
        hessian_approx = transform_hessian(
            triangle_vertices, barycentric_hessian_vectorized_approx
        )
        assert (hessian - hessian_approx).norm() < eps
