"""Test shape function code."""

import random
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
from tvemoves_rufbad.bell_finite_element import (
    bell_interpolation,
    bell_interpolation_gradient,
    bell_interpolation_hessian,
    bell_interpolation_on_edge,
)
from .helpers import random_polynomial_2d, random_barycentric_coordinates, point_c1_params


def test_cyclic_permutation() -> None:
    """Assure that cyclic permutation restores the original expression."""
    n_first_third_new = _N_first_third
    for _ in range(3):
        n_first_third_new = [_cyclic_permutation(ni, _L, _b, _c) for ni in n_first_third_new]
    assert all(ni_new == ni for ni_new, ni in zip(n_first_third_new, _N_first_third))


def test_independence_of_opposite_point() -> None:
    """Check that the shape function on an edge is independent of the point opposite to the edge."""
    for i in range(12, 18):
        assert _N[i].subs(_L[2], 0) == 0

    for i in range(12):
        assert _N[i].subs(_L[2], 0).free_symbols.issubset({_L[0], _L[1], _b[2], _c[2]})

    for ni in _N_on_edge:
        assert ni.free_symbols.issubset({_t, _b[2], _c[2]})


def test_bell_interpolation() -> None:
    """Assure that the shape function can recover a degree 4 polynomial in a triangle.

    see e.g. page 117 in https://people.sc.fsu.edu/~jburkardt/classes/fem_2011/chapter6.pdf
    """

    num_evaluations = 10
    eps = 1e-10

    triangle_vertices = (Vector([1.0, 2.0]), Vector([5.0, 2.4]), Vector([4.5, 8.2]))
    p1, p2, p3 = triangle_vertices

    poly, poly_gradient, poly_hessian = random_polynomial_2d(degree=4, num_derivatives=2)

    params1 = Vector(point_c1_params(p1, poly, poly_gradient, poly_hessian))
    params2 = Vector(point_c1_params(p2, poly, poly_gradient, poly_hessian))
    params3 = Vector(point_c1_params(p3, poly, poly_gradient, poly_hessian))
    params = params1.extend(params2).extend(params3)

    for c in random_barycentric_coordinates(num_evaluations):
        c_euclidean = c.l1 * p1 + c.l2 * p2 + c.l3 * p3
        value = poly(*c_euclidean)
        value_approx = bell_interpolation(triangle_vertices, c, params)
        assert abs(value - value_approx) < eps

        gradient = poly_gradient(*c_euclidean)
        gradient_approx = bell_interpolation_gradient(triangle_vertices, c, params)
        assert (gradient - gradient_approx).norm() < eps

        hessian = poly_hessian(*c_euclidean)
        hessian_approx = bell_interpolation_hessian(triangle_vertices, c, params)
        assert (hessian - hessian_approx).norm() < eps

    edge_vertices = (p1, p2)
    for _ in range(num_evaluations):
        t = random.random()
        p = t * p1 + (1 - t) * p2
        value = poly(*p)
        value_approx = bell_interpolation_on_edge(edge_vertices, t, params[:12])
        assert abs(value - value_approx) < eps
