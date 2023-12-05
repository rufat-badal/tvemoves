"""Test quadrature rules."""


import sympy as sp
from tvemoves_rufbad.quadrature_rules import (
    DUNAVANT_QUADRATURE_RULES,
    GaussQuadratureRule,
)
from .helpers import random_polynomial_symbolic_2d, random_polynomial_symbolic_1d


def test_dunavant() -> None:
    """Test that the dunavant quadrature rule of degree p exactly integrates polynomial
    of degree p.
    """

    eps = 1e-14

    for i, rule in enumerate(DUNAVANT_QUADRATURE_RULES):
        # rule i must integrate a polynomial of degree i + 1 exactly
        degree = i + 1
        poly_symbolic, [x, y] = random_polynomial_symbolic_2d(degree)
        poly_integral = sp.integrate(poly_symbolic, (y, 0, 1 - x), (x, 0, 1))
        poly = sp.lambdify([x, y], poly_symbolic)
        poly_quadrature = 0.5 * sum(w * poly(p[1], p[2]) for w, p in rule)
        assert abs(poly_quadrature - poly_integral) < eps


def test_gauss() -> None:
    """Test 1D Gauss quadrature. A quadrature rule of degree d should exactly integrate
    polynomials of degree d.
    """

    x = sp.Symbol("x")
    eps = 1e-14

    for degree in range(1, 6):
        rule = GaussQuadratureRule(degree)
        poly_symbolic, x = random_polynomial_symbolic_1d(degree)
        poly_integral = sp.integrate(poly_symbolic, (x, 0, 1))
        poly = sp.lambdify(x, poly_symbolic)
        poly_quadrature = sum(w * poly(p) for w, p in rule)
        assert abs(poly_integral - poly_quadrature) < eps
