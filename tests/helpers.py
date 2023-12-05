""""Helper functions used in some of the tests."""


import random
from typing import Callable
import sympy as sp
from tvemoves_rufbad.domain import BarycentricCoordinates
from tvemoves_rufbad.tensors import Vector, Matrix

Expr = sp.core.expr.Expr


def _jacobian_symbolic(f, symbols: list[sp.Symbol]):
    if isinstance(f, list):
        return [_jacobian_symbolic(component, symbols) for component in f]

    return [sp.diff(f, x) for x in symbols]


def random_polynomial_symbolic_2d(degree: int, num_derivatives: int = 0):
    """Generate symbolic expression of a random 2d polynomial and possibly some of
    its derivatives.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [
        [random.uniform(-10.0, 10.0) for _ in range(i + 1)] for i in range(degree + 1)
    ]

    x, y = sp.symbols("x y")
    poly = sum(
        c * x ** (i - j) * y**j
        for i, cs in enumerate(coefficients)
        for j, c in enumerate(cs)
    )

    if num_derivatives == 0:
        return poly, [x, y]

    random_poly_and_derivatives = [poly]
    cur_derivative = poly
    for _ in range(num_derivatives):
        cur_derivative = _jacobian_symbolic(cur_derivative, [x, y])
        random_poly_and_derivatives.append(cur_derivative)

    return random_poly_and_derivatives, [x, y]


Func2d = Callable[[float, float], any]


def random_polynomial_2d(
    degree: int, num_derivatives: int = 0
) -> Func2d | list[Func2d, ...]:
    """Generate a random 2d polynomial and possibly some of its derivatives.
    The coefficients are uniformly random in the interval (-10, 10).
    """

    random_poly_and_derivatives_symbolic, [x, y] = random_polynomial_symbolic_2d(
        degree, num_derivatives
    )

    if num_derivatives == 0:
        # random_poly_and_derivatives_symbolic is a single expression
        return sp.lambdify([x, y], random_poly_and_derivatives_symbolic)

    random_poly_and_derivatives = [
        sp.lambdify([x, y], f) for f in random_poly_and_derivatives_symbolic
    ]
    # Assure that the first derivatives returns a vector
    old_gradient = random_poly_and_derivatives[1]

    def new_gradient(x: float, y: float) -> Vector:
        return Vector(old_gradient(x, y))

    random_poly_and_derivatives[1] = new_gradient

    # Assure that the second derivatives (if it was requested) returns a matrix
    if num_derivatives > 1:
        old_hessian = random_poly_and_derivatives[2]

        def new_hessian(x: float, y: float) -> Matrix:
            return Matrix(old_hessian(x, y))

        random_poly_and_derivatives[2] = new_hessian

    return random_poly_and_derivatives


def random_polynomial_symbolic_1d(degree: int, num_derivatives: int = 0):
    """Generate symbolic expression of a random 1d polynomial and possibly some of its
    derivatives.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [random.uniform(-10.0, 10.0) for _ in range(degree + 1)]

    x = sp.Symbol("x")
    poly = sum(c * x**i for i, c in enumerate(coefficients))

    if num_derivatives == 0:
        return poly, x

    random_poly_and_derivatives = [poly]
    cur_derivative = poly
    for _ in range(num_derivatives):
        cur_derivative = sp.diff(cur_derivative, x)
        random_poly_and_derivatives.append(cur_derivative)

    return random_poly_and_derivatives, x


Func1d = Callable[float, any]


def random_polynomial_1d(
    degree: int, num_derivatives: int = 0
) -> Func1d | list[Func1d]:
    """Generate random 1d polynomial and possibly some of its derivatives.
    The coefficients are uniformly random in the interval (-10, 10).
    """
    random_poly_and_derivatives_symbolic, x = random_polynomial_symbolic_1d(
        degree, num_derivatives
    )

    return [sp.lambdify(x, f) for f in random_poly_and_derivatives_symbolic]


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
