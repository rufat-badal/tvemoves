""""Test quadrature rules and integrators."""

import sympy as sp
from .helpers import random_symbolic_polynomial
from tvemoves_rufbad.quadrature_rules import DUNAVANT_QUADRATURE_RULES


def test_dunavant() -> None:
    """Test that the dunavant quadrature rule of degree p exactly integrates polynomial
    of degree p.
    """

    x, y = sp.symbols("x y")
    eps = 1e-14

    for i, rule in enumerate(DUNAVANT_QUADRATURE_RULES):
        degree = i + 1
        poly_symbolic = random_symbolic_polynomial(degree, x, y)
        poly_integral = sp.integrate(poly_symbolic, (y, 0, 1 - x), (x, 0, 1))
        poly = sp.lambdify([x, y], poly_symbolic)
        poly_quadrature = 0.5 * sum(w * poly(p[1], p[2]) for w, p in rule)
        assert abs(poly_quadrature - poly_integral) < eps
