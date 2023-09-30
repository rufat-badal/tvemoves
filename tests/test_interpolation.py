from tvemoves_rufbad.interpolation import (
    P1Interpolation,
    P1Deformation,
    C1Interpolation,
)
from tvemoves_rufbad.grid import generate_square_equilateral_grid
from tvemoves_rufbad.tensors import Vector, Matrix
import pyomo.environ as pyo
from math import pi
from pytest import approx


def affine(x: float, y: float) -> float:
    return x + y


def gradient_affine(x: float, y: float) -> Vector:
    return Vector([1, 1])


def hessian_affine(x: float, y: float) -> Matrix:
    return Matrix([[0, 0], [0, 0]])


def parabola(x: float, y: float) -> float:
    return (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2


def gradient_parabola(x: float, y: float) -> Vector:
    return Vector([2 * (x - 1 / 2), 2 * (y - 1 / 2)])


def hessian_parabola(x: float, y: float) -> Matrix:
    return Matrix([[2, 0], [0, 2]])


def periodic(x: float, y: float) -> float:
    return pyo.sin(2 * pi * x) * pyo.cos(2 * pi * y)


def gradient_periodic(x: float, y: float) -> Vector:
    return Vector(
        [
            2 * pi * pyo.cos(2 * pi * y) * pyo.cos(2 * pi * x),
            -2 * pi * pyo.sin(2 * pi * x) * pyo.sin(2 * pi * y),
        ]
    )


def hessian_periodic(x: float, y: float) -> Matrix:
    return (
        -4
        * pi**2
        * Matrix(
            [
                [
                    pyo.cos(2 * pi * y) * pyo.sin(2 * pi * x),
                    pyo.sin(2 * pi * y) * pyo.cos(2 * pi * x),
                ],
                [
                    pyo.cos(2 * pi * x) * pyo.sin(2 * pi * y),
                    pyo.sin(2 * pi * x) * pyo.cos(2 * pi * y),
                ],
            ]
        )
    )


functions = [affine, parabola, periodic]
gradients = [gradient_affine, gradient_parabola, gradient_periodic]
hessians = [hessian_affine, hessian_parabola, hessian_periodic]


def affine_deformation(x: float, y: float) -> Vector:
    return Vector([2 * x, y])


def affine_strain(x: float, y: float) -> Matrix:
    return Matrix([[2, 0], [0, 1]])


def bend_deformation(x: float, y: float) -> Vector:
    angle = pi / 2 * x
    return (2 - y) * Vector([pyo.cos(angle), pyo.sin(angle)])


def bend_strain(x: float, y: float) -> Matrix:
    angle = pi / 2 * x
    return Matrix(
        [
            [-pi / 2 * (2 - y) * pyo.sin(angle), -pyo.cos(angle)],
            [pi / 2 * (2 - y) * pyo.cos(angle), -pyo.sin(angle)],
        ]
    )


def squeeze_deformation(x: float, y: float) -> Vector:
    return Vector([x - 1 / 2, 1 / 2 * (4 * x**2 + 1) * (y - 1 / 2)])


def squeeze_strain(x: float, y: float) -> Matrix:
    return Matrix([[1, 0], [4 * x * (y - 1 / 2), 1 / 2 * (4 * x**2 + 1)]])


deformations = [affine_deformation, bend_deformation, squeeze_deformation]
strains = [affine_strain, bend_strain, squeeze_strain]


def test_p1_interpolation() -> None:
    eps = 1e-6
    grad_eps = 1e-2
    grid = generate_square_equilateral_grid(num_horizontal_points=200)
    p0 = grid.points
    evaluation_points = [
        p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
    ]

    for f, grad_f in zip(functions, gradients):
        params = [f(p[0], p[1]) for p in grid.points]
        f_approx = P1Interpolation(grid, params)

        values = [f(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            f_approx(triangle, (1 / 3, 1 / 3, 1 / 3)) for triangle in grid.triangles
        ]
        mean_squared_error = sum(
            (value - value_approx) ** 2
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        grad_values = [grad_f(p[0], p[1]) for p in evaluation_points]
        grad_values_approx = [
            f_approx.gradient(triangle) for triangle in grid.triangles
        ]
        mean_squared_grad_error = sum(
            (grad_value - grad_value_approx).normsqr()
            for (grad_value, grad_value_approx) in zip(grad_values, grad_values_approx)
        ) / len(grid.triangles)
        assert mean_squared_grad_error < grad_eps


def test_p1_interpolation_with_pyomo_params() -> None:
    grid = generate_square_equilateral_grid(num_horizontal_points=50)

    for f in functions:
        params = [f(p[0], p[1]) for p in grid.points]
        f_approx = P1Interpolation(grid, params)
        values = [
            f_approx(triangle, (1 / 3, 1 / 3, 1 / 3)) for triangle in grid.triangles
        ]

        model = pyo.ConcreteModel()
        model.params = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=params,
            mutable=True,
        )
        f_approx_pyomo = P1Interpolation(grid, model.params)
        values_pyomo = [
            pyo.value(f_approx_pyomo(triangle, (1 / 3, 1 / 3, 1 / 3)))
            for triangle in grid.triangles
        ]
        assert values == approx(values_pyomo)

        grad_values = [f_approx.gradient(triangle)._data for triangle in grid.triangles]
        grad_values_pyomo = [
            f_approx_pyomo.gradient(triangle).map(pyo.value)._data
            for triangle in grid.triangles
        ]
        assert grad_values == grad_values_pyomo


def test_p1_deformation() -> None:
    eps = 1e-6
    grad_eps = 1e-3
    grid = generate_square_equilateral_grid(num_horizontal_points=100)
    p0 = grid.points
    evaluation_points = [
        p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
    ]

    for deform, strain in zip(deformations, strains):
        params_vectors = [deform(p[0], p[1]) for p in grid.points]
        params = (
            [v[0] for v in params_vectors],
            [v[1] for v in params_vectors],
        )
        deform_approx = P1Deformation(grid, *params)

        values = [deform(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            deform_approx(triangle, (1 / 3, 1 / 3, 1 / 3)).map(pyo.value)
            for triangle in grid.triangles
        ]
        mean_squared_error = sum(
            (value - value_approx).normsqr()
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        strain_values = [strain(p[0], p[1]) for p in evaluation_points]
        strain_values_approx = [
            deform_approx.strain(triangle) for triangle in grid.triangles
        ]
        mean_squared_strain_error = sum(
            (strain_value - strain_value_approx).normsqr()
            for (strain_value, strain_value_approx) in zip(
                strain_values, strain_values_approx
            )
        ) / len(grid.triangles)
        assert mean_squared_strain_error < grad_eps


def test_c1_interpolation() -> None:
    eps = 1e-6
    grad_eps = 1e-4
    grid = generate_square_equilateral_grid(num_horizontal_points=8)
    p0 = grid.points
    evaluation_points = [
        p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
    ]

    for f, grad_f, hessian_f in zip(functions, gradients, hessians):
        f_at_grid_points = [f(p[0], p[1]) for p in grid.points]
        grad_f_at_grid_points = [grad_f(p[0], p[1]) for p in grid.points]
        hessian_f_at_grid_points = [hessian_f(p[0], p[1]) for p in grid.points]
        params = [
            [f, G[0], G[1], H[0, 0], H[0, 1], H[1, 1]]
            for (f, G, H) in zip(
                f_at_grid_points, grad_f_at_grid_points, hessian_f_at_grid_points
            )
        ]
        f_approx = C1Interpolation(grid, params)

        values = [f(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            f_approx(triangle, (1 / 3, 1 / 3, 1 / 3)) for triangle in grid.triangles
        ]
        mean_squared_error = sum(
            (value - value_approx) ** 2
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        grad_values = [grad_f(p[0], p[1]) for p in evaluation_points]
        grad_values_approx = [
            f_approx.gradient(triangle, (1 / 3, 1 / 3, 1 / 3))
            for triangle in grid.triangles
        ]
        mean_squared_grad_error = sum(
            (grad_value - grad_value_approx).normsqr()
            for (grad_value, grad_value_approx) in zip(grad_values, grad_values_approx)
        ) / len(grid.triangles)
        assert mean_squared_grad_error < grad_eps


def f_5th_order(x: float, y: float) -> float:
    return x**5 + x**3 * y**2 + y**5


def gradient_f_5th_order(x: float, y: float) -> Vector:
    return Vector([5 * x**4 + 3 * x**2 * y**2, 2 * x**3 * y + 5 * y**4])


def hessian_f_5th_order(x: float, y: float) -> Matrix:
    return Matrix(
        [
            [20 * x**3 + 6 * x * y**2, 6 * x**2 * y],
            [6 * x**2 * y, 2 * x**3 + 20 * y**3],
        ]
    )


def test_c1_interpolation_5th_order() -> None:
    eps = 1e-6
    grid = generate_square_equilateral_grid(num_horizontal_points=3)
    p0 = grid.points
    evaluation_points = [
        p0[i1] / 3 + p0[i2] / 3 + p0[i3] / 3 for (i1, i2, i3) in grid.triangles
    ]

    f_at_grid_points = [f_5th_order(p[0], p[1]) for p in grid.points]
    grad_f_at_grid_points = [gradient_f_5th_order(p[0], p[1]) for p in grid.points]
    hessian_f_at_grid_points = [hessian_f_5th_order(p[0], p[1]) for p in grid.points]
    params = [
        [f, G[0], G[1], H[0, 0], H[0, 1], H[1, 1]]
        for (f, G, H) in zip(
            f_at_grid_points, grad_f_at_grid_points, hessian_f_at_grid_points
        )
    ]
    f_approx = C1Interpolation(grid, params)

    values = [f_5th_order(p[0], p[1]) for p in evaluation_points]
    values_approx = [
        f_approx(triangle, (1 / 3, 1 / 3, 1 / 3)) for triangle in grid.triangles
    ]
    mean_squared_error = sum(
        (value - value_approx) ** 2
        for (value, value_approx) in zip(values, values_approx)
    ) / len(grid.triangles)
    assert mean_squared_error < eps

    grad_values = [gradient_f_5th_order(p[0], p[1]) for p in evaluation_points]
    grad_values_approx = [
        f_approx.gradient(triangle, (1 / 3, 1 / 3, 1 / 3))
        for triangle in grid.triangles
    ]
    mean_squared_grad_error = sum(
        (grad_value - grad_value_approx).normsqr()
        for (grad_value, grad_value_approx) in zip(grad_values, grad_values_approx)
    ) / len(grid.triangles)
    assert mean_squared_grad_error < eps
