""""Test quadrature rules and integrators."""

import sympy as sp
from helpers import random_symbolic_polynomial
from tvemoves_rufbad.quadrature_rules import DUNAVANT_QUADRATURE_RULES


def test_dunavant() -> None:
    """Test that the dunavant quadrature rule of degree p exactly integrates polynomial
    of degree p.
    """

    x, y = sp.symbols("x y")

    rule = DUNAVANT_QUADRATURE_RULES[1]
    for degree in range(6):
        poly_symbolic = random_symbolic_polynomial(degree, x, y)
        poly_integral = sp.integrate(poly_symbolic, (y, 0, 1 - x), (x, 0, 1))
        print(poly_integral)

        poly = sp.lambdify([x, y], poly_symbolic)
        poly_quadrature = 0.5 * sum(
            w * poly(p[1], p[2]) for (w, p) in zip(rule.weights, rule.points)
        )
        print(poly_quadrature)

        print()


test_dunavant()
