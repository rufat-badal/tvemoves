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
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.bell_finite_element import shape_function
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

    see e.g.: https://people.sc.fsu.edu/~jburkardt/classes/fem_2011/chapter6.pdf
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

    parameters = []
    for v in triangle_vertices:
        parameters.extend(
            [
                poly(v[0], v[1]),
                poly_dx(v[0], v[1]),
                poly_dy(v[0], v[1]),
                poly_dxx(v[0], v[1]),
                poly_dxy(v[0], v[1]),
                poly_dyy(v[0], v[1]),
            ]
        )
    parameters = Vector(parameters)

    for c in random_barycentric_coordinates(num_evaluations):
        value_approx = shape_function(triangle_vertices, c).dot(parameters)
        c_euclidean = c.u * p1 + c.v * p2 + c.w * p3
        value = poly(*c_euclidean)
        assert abs(value - value_approx) < eps
