"""Computes the shape function and its derivatives for Bell's finite elements."""

# see also: https://doi.org/10.1002/nme.1620300303

import sympy as sp
from tvemoves_rufbad.tensors import Vector, Matrix
from tvemoves_rufbad.domain import (
    BarycentricCoordinates,
    TriangleVertices,
    EdgeVertices,
)


_L1, _L2, _L3 = sp.symbols("L1 L2 L3")
_L = [_L1, _L2, _L3]
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

_N_first_third = [_N1, _N2, _N3, _N4, _N5, _N6]


def _single_cyclic_permutation(
    expr: sp.core.expr.Expr, variable: list[sp.Symbol]
) -> sp.core.expr.Expr:
    if len(variable) <= 1:
        return expr

    s = sp.Symbol("s")
    replacements = [(variable[-1], s)]
    replacements.extend(
        [(variable[i - 1], variable[i]) for i in range(len(variable) - 1, 0, -1)]
    )
    replacements.append((s, variable[0]))
    return expr.subs(replacements)


def _cyclic_permutation(
    expr: sp.core.expr.Expr, *variables: list[sp.Symbol]
) -> sp.core.expr.Expr:
    for variable in variables:
        expr = _single_cyclic_permutation(expr, variable)
    return expr


_N_second_third = [_cyclic_permutation(ni, _L, _b, _c) for ni in _N_first_third]
_N_third_third = [_cyclic_permutation(ni, _L, _b, _c) for ni in _N_second_third]
_N = _N_first_third + _N_second_third + _N_third_third

_N_lambdified = sp.lambdify(_L + _b + _c, _N)


def _get_a(triangle_vertices: TriangleVertices) -> tuple[float, float, float]:
    p1, p2, p3 = triangle_vertices
    return (
        p2[0] * p3[1] - p3[0] * p2[1],
        p3[0] * p1[1] - p1[0] * p3[1],
        p1[0] * p2[1] - p2[0] * p1[1],
    )


def _get_b(triangle_vertices: TriangleVertices) -> tuple[float, float, float]:
    p1, p2, p3 = triangle_vertices
    return (p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1])


def _get_c(triangle_vertices: TriangleVertices) -> tuple[float, float, float]:
    p1, p2, p3 = triangle_vertices
    return (p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0])


def shape_function(
    triangle_vertices: TriangleVertices, barycentric_coordinates: BarycentricCoordinates
) -> Vector:
    """Shape function for a given triangle.

    Returns a vector of size 18.
    """
    return Vector(
        _N_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        )
    )


_N_jacobian = sp.Matrix(
    [[sp.diff(_N[i], _L[j]) for j in range(3)] for i in range(3 * 6)]
)
_N_jacobian_lambdified = sp.lambdify(_L + _b + _c, _N_jacobian)


def shape_function_jacobian(
    triangle_vertices: TriangleVertices,
    barycentric_coordinates: BarycentricCoordinates,
) -> Matrix:
    """Gradient of the symbolic shape function.

    Returns a matrix of shape (18,3).
    """
    return Matrix(
        _N_jacobian_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        ).tolist()
    )


_N_hessian = sp.Array(
    [
        [[sp.diff(_N[i], _L[j], _L[k]) for k in range(3)] for j in range(3)]
        for i in range(3 * 6)
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
    """Vectorized Hessian of the symbolic shape function.

    Returns a matrix of shape (18, 6).
    Row i of the matrix contains entries the Hessian matrix H (of shape (3, 3)) of
    the i-th component of the shape function in the following order:
    H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 2]
    """
    return Matrix(
        _N_hessian_vectorized_lambdified(
            *barycentric_coordinates,
            *_get_b(triangle_vertices),
            *_get_c(triangle_vertices),
        ).tolist()
    )


_t = sp.Symbol("t")
_N_on_edge = [_N[i].subs([(_L1, _t), (_L2, 1 - _t), (_L3, 0)]) for i in range(12)]
_N_on_edge_lambdified = sp.lambdify([_t, _b3, _c3], _N_on_edge)


def shape_function_on_edge(edge_vertices: EdgeVertices, t: float) -> Vector:
    """Shape function on an edge.

    Returns a vector of size 6.
    Note that the last 6 components of the full shape function vanish on the edge.
    """
    p1, p2 = edge_vertices
    b3 = p1[1] - p2[1]
    c3 = p2[0] - p1[0]
    return _N_on_edge_lambdified(t, b3, c3)


def transform_gradient(
    triangle_vertices: TriangleVertices,
    barycentric_gradient: Vector,
) -> Vector:
    """Transforms gradient with respect to barycentric coordinates to Euclidean gradient.

    barycentric_gradient should be a vector of size 3.
    Returns a vector of size 2.
    """
    a = _get_a(triangle_vertices)
    delta = sum(a) / 2
    b = _get_b(triangle_vertices)
    c = _get_c(triangle_vertices)
    trafo_matrix = Matrix([[b[0], b[1], b[2]], [c[0], c[1], c[2]]]) / (2 * delta)
    return trafo_matrix.dot(barycentric_gradient)


def transform_hessian(
    triangle_vertices: TriangleVertices,
    barycentric_hessian_vectorized: Vector,
) -> Matrix:
    """Transforms vectorized hessian with respect to barycentric coordinates to Euclidean
    vectorized hessian.

    barycentric_hessian_vectorized should be a vector of size 6 whose entries encode the
    barycentric Hessian H_bary (matrix of shape (3, 3)) in the following way:
    H_bary[0, 0], H_bary[1, 1], H_bary[2, 2], H_bary[0, 1], H_bary[0, 2], H_bary[1, 2]

    Returns the Euclidean Hessian H (matrix of shape (2, 2))
    """
    a = _get_a(triangle_vertices)
    delta = sum(a) / 2
    b = _get_b(triangle_vertices)
    c = _get_c(triangle_vertices)
    trafo_matrix = Matrix(
        [
            [
                b[0] ** 2,
                b[1] ** 2,
                b[2] ** 2,
                2 * b[0] * b[1],
                2 * b[0] * b[2],
                2 * b[1] * b[2],
            ],
            [
                c[0] ** 2,
                c[1] ** 2,
                c[2] ** 2,
                2 * c[0] * c[1],
                2 * c[0] * c[2],
                2 * c[1] * c[2],
            ],
            [
                b[0] * c[0],
                b[1] * c[1],
                b[2] * c[2],
                b[0] * c[1] + b[1] * c[0],
                b[0] * c[2] + b[2] * c[0],
                b[1] * c[2] + b[2] * c[1],
            ],
        ]
    ) / (4 * delta**2)
    hessian_vectorized = trafo_matrix.dot(barycentric_hessian_vectorized)
    return Matrix(
        [
            [hessian_vectorized[0], hessian_vectorized[2]],
            [hessian_vectorized[2], hessian_vectorized[1]],
        ]
    )
