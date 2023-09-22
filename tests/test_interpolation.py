from tvemoves_rufbad.interpolation import P1Interpolation
from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.tensors import Vector
import pyomo.environ as pyo
from pytest import approx

from math import sin, cos, pi


def affine(x, y):
    return x + y


def gradient_affine(x, y):
    return Vector([1, 1])


def parabola(x, y):
    return (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2


def gradient_parabola(x, y):
    return Vector([2 * (x - 1 / 2), 2 * (y - 1 / 2)])


def periodic(x, y):
    return sin(2 * pi * x) * cos(2 * pi * y)


def gradient_periodic(x, y):
    return Vector(
        [
            2 * pi * cos(2 * pi * y) * cos(2 * pi * x),
            -2 * pi * sin(2 * pi * x) * sin(2 * pi * y),
        ]
    )


functions = [affine, parabola, periodic]
gradients = [gradient_affine, gradient_parabola, gradient_periodic]


def test_p1_interpolation():
    eps = 1e-6
    grad_eps = 1e-2
    grid = SquareEquilateralGrid(num_horizontal_points=200)
    p0 = grid.initial_positions
    evaluation_points = [
        p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
    ]

    for f, grad_f in zip(functions, gradients):
        params = [f(*p) for p in grid.initial_positions]
        f_approx = P1Interpolation(grid, params)

        values = [f(*p) for p in evaluation_points]
        values_approx = [
            f_approx(triangle, (1 / 3, 1 / 3)) for triangle in grid.triangles
        ]
        mean_squared_error = sum(
            (value - value_approx) ** 2
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        grad_values = [grad_f(*p) for p in evaluation_points]
        grad_values_approx = [
            f_approx.gradient(triangle) for triangle in grid.triangles
        ]
        mean_squared_grad_error = sum(
            (grad_value - grad_value_approx).normsqr()
            for (grad_value, grad_value_approx) in zip(grad_values, grad_values_approx)
        ) / len(grid.triangles)
        print(mean_squared_grad_error)
        assert mean_squared_grad_error < grad_eps


def test_p1_interpolation_with_pyomo_params():
    grid = SquareEquilateralGrid(num_horizontal_points=2)

    for f in functions:
        params = [f(*p) for p in grid.initial_positions]
        f_approx = P1Interpolation(grid, params)
        values = [f_approx(triangle, (1 / 3, 1 / 3)) for triangle in grid.triangles]

        model = pyo.ConcreteModel()
        model.params = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=params,
            mutable=True,
        )
        f_approx_pyomo = P1Interpolation(grid, model.params)
        values_pyomo = [
            pyo.value(f_approx_pyomo(triangle, (1 / 3, 1 / 3)))
            for triangle in grid.triangles
        ]
        assert values == approx(values_pyomo)

        grad_values = [f_approx.gradient(triangle)._data for triangle in grid.triangles]
        grad_values_pyomo = [
            f_approx_pyomo.gradient(triangle).map(pyo.value)._data
            for triangle in grid.triangles
        ]
        assert grad_values == grad_values_pyomo
