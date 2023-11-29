""""Helper functions used in some of the tests."""

import sympy as sp
import random
from tvemoves_rufbad.domain import BarycentricCoordinates


def random_symbolic_polynomial_2d(
    degree: int, x: sp.Symbol, y: sp.Symbol
) -> sp.core.expr.Expr:
    """Generate symbolic expression of a random 2d polynomial of degree degree.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [
        [random.uniform(-10.0, 10.0) for _ in range(i + 1)] for i in range(degree + 1)
    ]

    poly = sum(
        c * x ** (i - j) * y**j
        for i, cs in enumerate(coefficients)
        for j, c in enumerate(cs)
    )

    return poly


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
