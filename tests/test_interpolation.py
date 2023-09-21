from tvemoves_rufbad.interpolation import P1Interpolation
from tvemoves_rufbad.grid import SquareEquilateralGrid
from math import sin, cos, pi


def affine(x, y):
    return x + y


def parabola(x, y):
    return (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2


def periodic(x, y):
    return sin(2 * pi * x) * cos(2 * pi * y)


functions = [affine, parabola, periodic]

EPS = 1.0e-6


def test_p1_interpolation():
    max_steps = 10
    for f in functions:
        f_approximated = False
        num_horizontal_points = 1
        for _ in range(max_steps):
            num_horizontal_points *= 2
            grid = SquareEquilateralGrid(num_horizontal_points)
            params = [f(*p) for p in grid.initial_positions]
            f_approx = P1Interpolation(grid, params)
            p0 = grid.initial_positions
            evaluation_points = [
                p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
            ]
            values = [f(*p) for p in evaluation_points]
            values_approx = [
                f_approx(triangle, (1 / 3, 1 / 3)) for triangle in grid.triangles
            ]
            mean_squared_error = sum(
                (value - value_approx) ** 2
                for (value, value_approx) in zip(values, values_approx)
            ) / len(grid.triangles)
            if mean_squared_error < EPS:
                f_approximated = True
                break
        assert f_approximated
