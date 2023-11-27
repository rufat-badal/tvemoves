"""Computes the shape function and its derivatives for Bell's finite elements."""

# see also: https://doi.org/10.1002/nme.1620300303

from typing import Iterable
import sympy as sp
from tvemoves_rufbad.tensors import Vector, Matrix
from tvemoves_rufbad.domain import BarycentricCoordinates, TriangleVertices


_L1, _L2, _L3 = sp.symbols("L1 L2 L3")
_L = [_L1, _L2, _L3]
_t = sp.symbols("t")
_b1, _b2, _b3 = sp.symbols("b1 b2 b3")
_b = [_b1, _b2, _b3]
_c1, _c2, _c3 = sp.symbols("c1 c2 c3")
_c = [_c1, _c2, _c3]
_r = sp.Matrix(
    [
        [-(_b[i] * _b[j] + _c[i] * _c[j]) / (_b[i] ** 2 + _c[i] ** 2) for j in range(3)]
        for i in range(3)
    ]
)
_r21 = _r[1, 0]
_r31 = _r[2, 0]

_N1 = (
    _L1**5
    + 5 * _L1**4 * _L2
    + 5 * _L1**4 * _L3
    + 10 * _L1**3 * _L2**2
    + 10 * _L1**3 * _L3**2
    + 20 * _L1**3 * _L2 * _L3
    + 30 * _r21 * _L1**2 * _L2 * _L3**2
    + 30 * _r31 * _L1**2 * _L3 * _L2**2
)
_N2 = (
    _c3 * _L1**4 * _L2
    - _c2 * _L1**4 * _L3
    + 4 * _c3 * _L1**3 * _L2**2
    - 4 * _c2 * _L1**3 * _L3**2
    + 4 * (_c3 - _c2) * _L1**3 * _L2 * _L3
    - (3 * _c1 + 15 * _r21 * _c2) * _L1**2 * _L2 * _L3**2
    + (3 * _c1 + 15 * _r31 * _c3) * _L1**2 * _L3 * _L2**2
)
_N3 = (
    -_b3 * _L1**4 * _L2
    + _b2 * _L1**4 * _L3
    - 4 * _b3 * _L1**3 * _L2**2
    + 4 * _b2 * _L1**3 * _L3**2
    + 4 * (_b2 - _b3) * _L1**3 * _L2 * _L3
    + (3 * _b1 + 15 * _r21 * _b2) * _L1**2 * _L2 * _L3**2
    - (3 * _b1 + 15 * _r31 * _b3) * _L1**2 * _L3 * _L2**2
)
_N4 = (
    _c3**2 / 2 * _L1**3 * _L2**2
    + _c2**2 / 2 * _L1**3 * _L3**2
    - _c2 * _c3 * _L1**3 * _L2 * _L3
    + (_c1 * _c2 + 5 / 2 * _r21 * _c2**2) * _L2 * _L3**2 * _L1**2
    + (_c1 * _c3 + 5 / 2 * _r31 * _c3**2) * _L3 * _L2**2 * _L1**2
)
_N5 = (
    -_b3 * _c3 * _L1**3 * _L2**2
    - _b2 * _c2 * _L1**3 * _L3**2
    + (_b2 * _c3 + _b3 * _c2) * _L1**3 * _L2 * _L3
    - (_b1 * _c2 + _b2 * _c1 + 5 * _r21 * _b2 * _c2) * _L2 * _L3**2 * _L1**2
    - (_b1 * _c3 + _b3 * _c1 + 5 * _r31 * _b3 * _c3) * _L3 * _L2**2 * _L1**2
)
_N6 = (
    _b3**2 / 2 * _L1**3 * _L2**2
    + _b2**2 / 2 * _L1**3 * _L3**2
    - _b2 * _b3 * _L1**3 * _L2 * _L3
    + (_b1 * _b2 + 5 / 2 * _r21 * _b2**2) * _L2 * _L3**2 * _L1**2
    + (_b1 * _b3 + 5 / 2 * _r31 * _b3**2) * _L3 * _L2**2 * _L1**2
)

_N = [_N1, _N2, _N3, _N4, _N5, _N6]

_N_lambdified = sp.lambdify(_L + _b + _c, _N)


def _get_b(triangle_vertices: TriangleVertices) -> tuple[float, float, float]:
    p1, p2, p3 = triangle_vertices
    return (p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1])


def _get_c(triangle_vertices: TriangleVertices) -> tuple[float, float, float]:
    p1, p2, p3 = triangle_vertices
    return (p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0])


def shape_function(
    triangle_vertices: TriangleVertices, barycentric_coordinates: BarycentricCoordinates
) -> Vector:
    """Shape function for a given triangle."""
    return Vector(
        _N_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        )
    )


print(
    shape_function(
        (Vector([0.0, 0.0]), Vector([1.0, 0.0]), Vector([1.0, 1.0])),
        (1 / 3, 1 / 3, 1 / 3),
    )
)


_N_jacobian = sp.Matrix([[sp.diff(_N[i], _L[j]) for j in range(3)] for i in range(6)])
_N_jacobian_lambdified = sp.lambdify(_L + _b + _c, _N_jacobian)


def shape_function_jacobian(
    triangle_vertices: TriangleVertices,
    barycentric_coordinates: BarycentricCoordinates,
) -> Matrix:
    """Lambdification of the gradient of the symbolic shape function."""
    return Matrix(
        _N_jacobian_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        ).tolist()
    )


print(
    shape_function_jacobian(
        (Vector([0.0, 0.0]), Vector([1.0, 0.0]), Vector([1.0, 1.0])),
        (1 / 3, 1 / 3, 1 / 3),
    )
)

_N_hessian = sp.Array(
    [
        [[sp.diff(_N[i], _L[j], _L[k]) for k in range(3)] for j in range(3)]
        for i in range(6)
    ]
)
_N_hessian_vectorized = sp.Matrix(
    [[H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 2]] for H in _N_hessian]
)
_N_hessian_vectorized_lambdified = sp.lambdify(_L + _b + _c, _N_hessian_vectorized)


def shape_function_hessian_vectorized(
    triangle_vertices: TriangleVertices,
    barycentric_coordinates: BarycentricCoordinates,
) -> Matrix:
    """Lambdification of the hessian of the symbolic shape function."""
    return Matrix(
        _N_hessian_vectorized_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        ).tolist()
    )


print(
    shape_function_hessian_vectorized(
        (Vector([0.0, 0.0]), Vector([1.0, 0.0]), Vector([1.0, 1.0])),
        (1 / 3, 1 / 3, 1 / 3),
    )
)

# _l1_t = sp.symbols("L1_t")
# _shape_function_on_edge_left_symbolic = [
#     Ni.subs(_L1, _l1_t).subs(_L2, 1 - _l1_t).subs(_L3, 0)
#     for Ni in _shape_function_symbolic
# ]
# _shape_function_on_edge_left_lambdified = sp.lambdify(
#     [_l1_t, _b3, _c3], _shape_function_on_edge_left_symbolic
# )


# def shape_function_on_edge_left(l1_on_edge: float, b3: float, c3: float):
#     """Lambdification of the symbolic shape function on an edge for the left point."""
#     return Vector(_shape_function_on_edge_left_lambdified(l1_on_edge, b3, c3))


# _shape_function_on_edge_right_symbolic = [
#     Ni.subs(_L1, 1 - _l1_t).subs(_L2, 0).subs(_L3, _l1_t)
#     for Ni in _shape_function_symbolic
# ]
# _shape_function_on_edge_right_lambdified = sp.lambdify(
#     [_l1_t, _b2, _c2], _shape_function_on_edge_right_symbolic
# )


# def shape_function_on_edge_right(l1_on_edge: float, b2: float, c2: float):
#     """Lambdification of the symbolic shape function on an edge for the right point."""
#     return Vector(_shape_function_on_edge_right_lambdified(l1_on_edge, b2, c2))
