""""Helper functions used in some of the tests."""

import sympy as sp
import random
from tvemoves_rufbad.domain import BarycentricCoordinates
from typing import Callable

Expr = sp.core.expr.Expr


def _jacobian_symbolic(f: Expr):
    if isinstance(f, list):
        return [_jacobian_symbolic(component) for component in f]
    else:
        return [sp.diff(f, x) for x in f.free_symbols]


def random_polynomial_2d(
    degree: int, symbols: tuple[sp.Symbol, sp.Symbol], num_derivatives: int = 0
) -> sp.core.expr.Expr:
    """Generate symbolic expression of a random 2d polynomial of degree degree.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [
        [random.uniform(-10.0, 10.0) for _ in range(i + 1)] for i in range(degree + 1)
    ]

    x, y = symbols
    poly = sum(
        c * x ** (i - j) * y**j
        for i, cs in enumerate(coefficients)
        for j, c in enumerate(cs)
    )

    random_poly_and_derivatives_symbolic = [poly]
    grad_poly = _derivative_2d_symbolic(poly, x, y)
    print(grad_poly)
    hessian_poly = _derivative_2d_symbolic(grad_poly, x, y)
    print(hessian_poly)

    return poly


x, y = sp.symbols("x y")
random_polynomial_2d(2, x, y, 1)


def random_symbolic_polynomial_1d(degree: int, x: sp.Symbol) -> sp.core.expr.Expr:
    """Generate symbolic expression of a random 1d polynomial of degree degree.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [random.uniform(-10.0, 10.0) for _ in range(degree + 1)]

    poly = sum(c * x**i for i, c in enumerate(coefficients))

    return poly


def random_barycentric_coordinates(
    num_coordinates: int,
) -> list[BarycentricCoordinates]:
    """Creates a list of uniformly sampled barycentric coordinates."""
    coordinates: list[BarycentricCoordinates] = []

    if num_coordinates < 0:
        return coordinates

    for _ in range(num_coordinates):
        x, y = sorted([random.random(), random.random()])
        u, v = y - x, x
        coordinates.append(BarycentricCoordinates(u, v))

    return coordinates
