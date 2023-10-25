"""Test integrators."""

from typing import Callable
from math import pi, isclose
import pyomo.environ as pyo
from tvemoves_rufbad.integrators import Integrator, BoundaryIntegrator
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.grid import (
    SquareEquilateralGrid,
    Triangle,
    Edge,
    BarycentricCoordinates,
)
from tvemoves_rufbad.quadrature_rules import TRIANGLE_QUADRATURE_RULES, DUNAVANT5

GAUSS_DEGREES = [1, 2, 3, 4, 5, 5, 7]


def constant(x: float, y: float) -> float:
    """Constant 1 function."""
    del x
    del y
    return 1


INT_CONSTANT = 1


def parabola(x: float, y: float) -> float:
    """2-d parabola"""
    return x**2 + y**2


INT_PARABOLA = 2 / 3


def periodic(x: float, y: float) -> float:
    """2-d periodic function."""
    return pyo.sin(2 * pi * x) * pyo.cos(2 * pi * y)


INT_PERIODIC = 0


def generate_integrand(
    f: Callable[[float, float], float], grid_points: list[Vector]
) -> Callable[[Triangle, BarycentricCoordinates], float]:
    """Given a function on the unit square, create integrand for the integrator."""

    def f_integrand(
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> float:
        i1, i2, i3 = triangle
        t1, t2, t3 = barycentric_coordinates
        p = t1 * grid_points[i1] + t2 * grid_points[i2] + t3 * grid_points[i3]
        return f(p[0], p[1])

    return f_integrand


functions = [constant, parabola, periodic]
integral_values = [INT_CONSTANT, INT_PARABOLA, INT_PERIODIC]


def constant_boundary(t: float) -> float:
    """Constant 1 function on the unit interval."""
    del t
    return 1


INT_CONST_BOUNDARY = 1


def polynomial_boundary(t: float) -> float:
    """Degree 8 polynomial on the unit interval."""
    return 10000 * (t - 1 / 2) ** 8


INT_POLYNOMIAL_BOUNDARY = 625 / 144


def periodic_boundary(t: float) -> float:
    """Sinusoidal function on the unit interval."""
    return 100 * pyo.sin(pi * t)


INT_PERIODIC_BOUNDARY = 200 / pi

BOUNDARY_FUNCTIONS = [constant_boundary, polynomial_boundary, periodic_boundary]
BOUNDARY_INTEGRAL_VALUES = [
    INT_CONST_BOUNDARY,
    INT_POLYNOMIAL_BOUNDARY,
    INT_PERIODIC_BOUNDARY,
]

EPS = 1e-5


def generate_edges_in_unit_interval(
    num_edges: int,
) -> tuple[list[tuple[int, int]], list[Vector]]:
    """Split unit interval into num_edges edges."""
    edges = [(i, i + 1) for i in range(num_edges)]
    # boundary integrator can only work with vectors
    points = [Vector([i / num_edges, 0]) for i in range(num_edges + 1)]
    return edges, points


def generate_boundary_integrand(
    f: Callable[[float], float], points: list[Vector]
) -> Callable[[Edge, float], float]:
    """Given a map on the unit interval create integrand for the Gauss integrator."""

    # f must be a scalar function on the unit interval
    def f_boundary(edge: Edge, t: float) -> float:
        i1, i2 = edge
        # only use first component (the second one is zero)
        return f(t * points[i1][0] + (1 - t) * points[i2][0])

    return f_boundary


def test_integrator() -> None:
    """Test triangle integrator with all available quadrature rules."""
    max_horizontal_points = 256
    for quadrature in TRIANGLE_QUADRATURE_RULES:
        for f, integral_value in zip(functions, integral_values):
            num_horizontal_points = 2
            approximation_converges = False
            while num_horizontal_points <= max_horizontal_points:
                grid = SquareEquilateralGrid(num_horizontal_points)
                integrand = generate_integrand(f, grid.points)
                integrator = Integrator(quadrature, grid.triangles, grid.points)
                error = abs(integrator(integrand) - integral_value)
                if error < EPS:
                    approximation_converges = True
                    break
                num_horizontal_points *= 2
            assert approximation_converges


def test_integrator_with_pyomo_parameters() -> None:
    """Check that the integrator also works with pyomo parameters instead of floats."""
    quadrature = DUNAVANT5
    grid = SquareEquilateralGrid(num_horizontal_points=30)
    integrator = Integrator(quadrature, grid.triangles, grid.points)
    for f in functions:
        integrand = generate_integrand(f, grid.points)
        model = pyo.ConcreteModel()
        model.initial_x1 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[0] for p in grid.points],
            mutable=True,
        )
        model.initial_x2 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[1] for p in grid.points],
            mutable=True,
        )
        points_pyomo = [
            Vector([model.initial_x1[i], model.initial_x2[i]]) for i in grid.vertices
        ]
        integrand_pyomo = generate_integrand(f, points_pyomo)
        assert isclose(integrator(integrand), pyo.value(integrator(integrand_pyomo)))


def test_boundary_integrator() -> None:
    """Test the boundary integrator."""
    max_points = 512
    for degree in range(2, 8):
        for f, integral_value in zip(BOUNDARY_FUNCTIONS, BOUNDARY_INTEGRAL_VALUES):
            num_points = 2
            approximation_converges = False
            while num_points <= max_points:
                edges, points = generate_edges_in_unit_interval(
                    num_edges=num_points - 1
                )
                integrand = generate_boundary_integrand(f, points)
                integrator = BoundaryIntegrator(degree, edges, points)
                error = abs(integrator(integrand) - integral_value)
                if error < EPS:
                    approximation_converges = True
                    break
                num_points *= 2
            assert approximation_converges
