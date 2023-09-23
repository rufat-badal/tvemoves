from tvemoves_rufbad.integrators import Integrator
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.quadrature_rules import (
    CENTROID,
    VERTEX,
    DUNAVANT2,
    DUNAVANT3,
    DUNAVANT4,
    DUNAVANT5,
)
import pyomo.environ as pyo
from math import pi, isclose

TRIANGLE_QUADRATURE_RULES = [
    CENTROID,
    VERTEX,
    DUNAVANT2,
    DUNAVANT3,
    DUNAVANT4,
    DUNAVANT5,
]

GAUSS_DEGREES = [1, 2, 3, 4, 5, 5, 7]


def constant(x, y):
    return 1


INT_CONSTANT = 1


def parabola(x, y):
    return x**2 + y**2


INT_PARABOLA = 2 / 3


def periodic(x, y):
    return pyo.sin(2 * pi * x) * pyo.cos(2 * pi * y)


INT_PERIODIC = 0


def generate_integrand(f, grid_points):
    def f_integrand(triangle, barycentric_coordinates):
        i1, i2, i3 = triangle
        b1, b2 = barycentric_coordinates
        b3 = 1 - b1 - b2
        p = b1 * grid_points[i1] + b2 * grid_points[i2] + b3 * grid_points[i3]
        return f(*p)

    return f_integrand


functions = [constant, parabola, periodic]
integral_values = [INT_CONSTANT, INT_PARABOLA, INT_PERIODIC]

EPS = 1e-5


def test_integrator():
    max_horizontal_points = 256
    for quadrature in TRIANGLE_QUADRATURE_RULES:
        for f, integral_value in zip(functions, integral_values):
            num_horizontal_points = 2
            approximation_converges = False
            while num_horizontal_points <= max_horizontal_points:
                grid = SquareEquilateralGrid(num_horizontal_points)
                integrand = generate_integrand(f, grid.initial_positions)
                integrator = Integrator(quadrature, grid)
                error = abs(integrator(integrand) - integral_value)
                if error < EPS:
                    approximation_converges = True
                    break
                num_horizontal_points *= 2
            print(error)
            assert approximation_converges


def test_integrator_with_pyomo_parameters():
    quadrature = DUNAVANT5
    grid = SquareEquilateralGrid(num_horizontal_points=30)
    integrator = Integrator(quadrature, grid)
    for f, integral_value in zip(functions, integral_values):
        integrand = generate_integrand(f, grid.initial_positions)
        model = pyo.ConcreteModel()
        model.initial_x1 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[0] for p in grid.initial_positions],
            mutable=True,
        )
        model.initial_x2 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[1] for p in grid.initial_positions],
            mutable=True,
        )
        initial_positions_pyomo = [
            Vector([model.initial_x1[i], model.initial_x2[i]]) for i in grid.vertices
        ]
        integrand_pyomo = generate_integrand(f, initial_positions_pyomo)
        assert isclose(integrator(integrand), pyo.value(integrator(integrand_pyomo)))
