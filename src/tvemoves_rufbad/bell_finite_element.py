"""Computes the shape function and its derivatives for Bell's finite elements."""

from typing import Iterable
import sympy as sp
from tvemoves_rufbad.tensors import Vector, Matrix


_l = list(sp.symbols("L1 L2 L3"))
_t = sp.symbols("t")
_b = list(sp.symbols("b1 b2 b3"))
_c = list(sp.symbols("c1 c2 c3"))
_r = sp.Matrix(
    [
        [-(_b[i] * _b[j] + _c[i] * _c[j]) / (_b[i] ** 2 + _c[i] ** 2) for j in range(3)]
        for i in range(3)
    ]
)

_shape_function_symbolic = []
_shape_function_symbolic.append(
    _l[0] ** 5
    + 5 * _l[0] ** 4 * _l[1]
    + 5 * _l[0] ** 4 * _l[2]
    + 10 * _l[0] ** 3 * _l[1] ** 2
    + 10 * _l[0] ** 3 * _l[2] ** 2
    + 20 * _l[0] ** 3 * _l[1] * _l[2]
    + 30 * _r[1, 0] * _l[0] ** 2 * _l[1] * _l[2] ** 2
    + 30 * _r[2, 0] * _l[0] ** 2 * _l[2] * _l[1] ** 2
)
_shape_function_symbolic.append(
    _c[2] * _l[0] ** 4 * _l[1]
    - _c[1] * _l[0] ** 4 * _l[2]
    + 4 * _c[2] * _l[0] ** 3 * _l[1] ** 2
    - 4 * _c[1] * _l[0] ** 3 * _l[2] ** 2
    + 4 * (_c[2] - _c[1]) * _l[0] ** 3 * _l[1] * _l[2]
    - (3 * _c[0] + 15 * _r[1, 0] * _c[1]) * _l[0] ** 2 * _l[1] * _l[2] ** 2
    + (3 * _c[0] + 15 * _r[2, 0] * _c[2]) * _l[0] ** 2 * _l[2] * _l[1] ** 2
)
_shape_function_symbolic.append(
    -_b[2] * _l[0] ** 4 * _l[1]
    + _b[1] * _l[0] ** 4 * _l[2]
    - 4 * _b[2] * _l[0] ** 3 * _l[1] ** 2
    + 4 * _b[1] * _l[0] ** 3 * _l[2] ** 2
    + 4 * (_b[1] - _b[2]) * _l[0] ** 3 * _l[1] * _l[2]
    + (3 * _b[0] + 15 * _r[1, 0] * _b[1]) * _l[0] ** 2 * _l[1] * _l[2] ** 2
    - (3 * _b[0] + 15 * _r[2, 0] * _b[2]) * _l[0] ** 2 * _l[2] * _l[1] ** 2
)
_shape_function_symbolic.append(
    _c[2] ** 2 / 2 * _l[0] ** 3 * _l[1] ** 2
    + _c[1] ** 2 / 2 * _l[0] ** 3 * _l[2] ** 2
    - _c[1] * _c[2] * _l[0] ** 3 * _l[1] * _l[2]
    + (_c[0] * _c[1] + 5 / 2 * _r[1, 0] * _c[1] ** 2) * _l[1] * _l[2] ** 2 * _l[0] ** 2
    + (_c[0] * _c[2] + 5 / 2 * _r[2, 0] * _c[2] ** 2) * _l[2] * _l[1] ** 2 * _l[0] ** 2
)
_shape_function_symbolic.append(
    -_b[2] * _c[2] * _l[0] ** 3 * _l[1] ** 2
    - _b[1] * _c[1] * _l[0] ** 3 * _l[2] ** 2
    + (_b[1] * _c[2] + _b[2] * _c[1]) * _l[0] ** 3 * _l[1] * _l[2]
    - (_b[0] * _c[1] + _b[1] * _c[0] + 5 * _r[1, 0] * _b[1] * _c[1])
    * _l[1]
    * _l[2] ** 2
    * _l[0] ** 2
    - (_b[0] * _c[2] + _b[2] * _c[0] + 5 * _r[2, 0] * _b[2] * _c[2])
    * _l[2]
    * _l[1] ** 2
    * _l[0] ** 2
)
_shape_function_symbolic.append(
    _b[2] ** 2 / 2 * _l[0] ** 3 * _l[1] ** 2
    + _b[1] ** 2 / 2 * _l[0] ** 3 * _l[2] ** 2
    - _b[1] * _b[2] * _l[0] ** 3 * _l[1] * _l[2]
    + (_b[0] * _b[1] + 5 / 2 * _r[1, 0] * _b[1] ** 2) * _l[1] * _l[2] ** 2 * _l[0] ** 2
    + (_b[0] * _b[2] + 5 / 2 * _r[2, 0] * _b[2] ** 2) * _l[2] * _l[1] ** 2 * _l[0] ** 2
)

_shape_function_lambdified = sp.lambdify(_l + _b + _c, _shape_function_symbolic)


def shape_function(
    area_coordinates: Iterable[float],
    b: Iterable[float],
    c: Iterable[float],
) -> Vector:
    """Lambdification of the symbolic shape function."""
    return Vector(_shape_function_lambdified(*area_coordinates, *b, *c))


_shape_function_jacobian_symbolic = sp.Matrix(
    [[sp.diff(_shape_function_symbolic[i], _l[j]) for j in range(3)] for i in range(6)]
)
_shape_function_jacobian_lambdified = sp.lambdify(
    _l + _b + _c, _shape_function_jacobian_symbolic
)


def shape_function_jacobian(
    area_coordinates: Iterable[float],
    b: Iterable[float],
    c: Iterable[float],
) -> Matrix:
    """Lambdification of the gradient of the symbolic shape function."""
    return Matrix(
        _shape_function_jacobian_lambdified(*area_coordinates, *b, *c).tolist()
    )


_shape_function_hessian_symbolic = sp.Array(
    [
        [
            [sp.diff(_shape_function_symbolic[i], _l[j], _l[k]) for k in range(3)]
            for j in range(3)
        ]
        for i in range(6)
    ]
)
_shape_function_hessian_vectorized_symbolic = sp.Matrix(
    [
        [H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 2]]
        for H in _shape_function_hessian_symbolic
    ]
)
_shape_function_hessian_vectorized_lambdified = sp.lambdify(
    _l + _b + _c, _shape_function_hessian_vectorized_symbolic
)


def shape_function_hessian_vectorized(
    area_coordinates: Iterable[float],
    b: Iterable[float],
    c: Iterable[float],
) -> Matrix:
    """Lambdification of the hessian of the symbolic shape function."""
    return Matrix(
        _shape_function_hessian_vectorized_lambdified(
            *area_coordinates, *b, *c
        ).tolist()
    )


_l1_t = sp.symbols("L1_t")
_shape_function_on_edge_left_symbolic = [
    Ni.subs(_l[0], _l1_t).subs(_l[1], 1 - _l1_t).subs(_l[2], 0)
    for Ni in _shape_function_symbolic
]
_shape_function_on_edge_left_lambdified = sp.lambdify(
    [_l1_t, _b[2], _c[2]], _shape_function_on_edge_left_symbolic
)


def shape_function_on_edge_left(l1_on_edge: float, b3: float, c3: float):
    """Lambdification of the symbolic shape function on an edge for the left point."""
    return Vector(_shape_function_on_edge_left_lambdified(l1_on_edge, b3, c3))


_shape_function_on_edge_right_symbolic = [
    Ni.subs(_l[0], 1 - _l1_t).subs(_l[1], 0).subs(_l[2], _l1_t)
    for Ni in _shape_function_symbolic
]
_shape_function_on_edge_right_lambdified = sp.lambdify(
    [_l1_t, _b[1], _c[1]], _shape_function_on_edge_right_symbolic
)


def shape_function_on_edge_right(l1_on_edge: float, b2: float, c2: float):
    """Lambdification of the symbolic shape function on an edge for the right point."""
    return Vector(_shape_function_on_edge_right_lambdified(l1_on_edge, b2, c2))
