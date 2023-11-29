import sympy as sp
import random


def random_symbolic_polynomial_2d(
    degree: int, x: sp.Symbol, y: sp.Symbol
) -> sp.core.expr.Expr:
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [
        [random.uniform(-10.0, 10.0) for _ in range(i + 1)] for i in range(degree + 1)
    ]

    poly = 0
    for i, cs in enumerate(coefficients):
        for j, c in enumerate(cs):
            poly += c * x ** (i - j) * y**j

    return poly


def random_symbolic_polynomial_1d(degree: int, x: sp.Symbol) -> sp.core.expr.Expr:
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coefficients = [random.uniform(-10.0, 10.0) for _ in range(degree + 1)]

    poly = 0
    for i, c in enumerate(coefficients):
        poly += c * x**i

    return poly
