"""Test interpolators."""

from math import pi
import pyomo.environ as pyo
from pytest import approx
import numpy as np
from tvemoves_rufbad.interpolation import (
    P1Interpolation,
    P1Deformation,
    C1Interpolation,
    C1Deformation,
)
from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D


def affine(x: float, y: float) -> float:
    """Affine map on the unit square."""
    return x + y


def gradient_affine(x: float, y: float) -> Vector:
    """Gradient of the affine map."""
    del x
    del y
    return Vector([1, 1])


def hessian_affine(x: float, y: float) -> Matrix:
    """Hessian of the affine map."""
    del x
    del y
    return Matrix([[0, 0], [0, 0]])


def parabola(x: float, y: float) -> float:
    """Parabola on the unit suqare."""
    return (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2


def gradient_parabola(x: float, y: float) -> Vector:
    """Gradient of the parabola."""
    return Vector([2 * (x - 1 / 2), 2 * (y - 1 / 2)])


def hessian_parabola(x: float, y: float) -> Matrix:
    """Hessian of the parabola."""
    del x
    del y
    return Matrix([[2, 0], [0, 2]])


def periodic(x: float, y: float) -> float:
    """Periodic function on the unit squre."""
    return pyo.sin(2 * pi * x) * pyo.cos(2 * pi * y)


def gradient_periodic(x: float, y: float) -> Vector:
    """Gradient of the periodic function."""
    return Vector(
        [
            2 * pi * pyo.cos(2 * pi * y) * pyo.cos(2 * pi * x),
            -2 * pi * pyo.sin(2 * pi * x) * pyo.sin(2 * pi * y),
        ]
    )


def hessian_periodic(x: float, y: float) -> Matrix:
    """Hessian of the periodic function."""
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
    """Affine deformation."""
    return Vector([2 * x, y])


def affine_strain(x: float, y: float) -> Matrix:
    """Strain of the affine deformation."""
    del x
    del y
    return Matrix([[2, 0], [0, 1]])


def affine_hyper_strain(x: float, y: float) -> Tensor3D:
    """Hyper strain of the affine deformation."""
    del x
    del y
    return Tensor3D([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])


def bend_deformation(x: float, y: float) -> Vector:
    """Deformation circularly bending the unit square."""
    angle = pi / 2 * x
    return (2 - y) * Vector([pyo.cos(angle), pyo.sin(angle)])


def bend_strain(x: float, y: float) -> Matrix:
    """Strain of the circular bending."""
    angle = pi / 2 * x
    return Matrix(
        [
            [-pi / 2 * (2 - y) * pyo.sin(angle), -pyo.cos(angle)],
            [pi / 2 * (2 - y) * pyo.cos(angle), -pyo.sin(angle)],
        ]
    )


def bend_hyper_strain(x: float, y: float) -> Tensor3D:
    """Hyper strain of the circular bending."""
    angle = pi / 2 * x
    return Tensor3D(
        [
            [
                [-(pi**2) / 4 * (2 - y) * pyo.cos(angle), pi / 2 * pyo.sin(angle)],
                [pi / 2 * pyo.sin(angle), 0],
            ],
            [
                [-(pi**2) / 4 * (2 - y) * pyo.sin(angle), -pi / 2 * pyo.cos(angle)],
                [-pi / 2 * pyo.cos(angle), 0],
            ],
        ]
    )


def squeeze_deformation(x: float, y: float) -> Vector:
    """Deformation squeezing the unit square horizontally."""
    return Vector([x - 1 / 2, 1 / 2 * (4 * x**2 + 1) * (y - 1 / 2)])


def squeeze_strain(x: float, y: float) -> Matrix:
    """Strain of the horizontal squeeze."""
    return Matrix([[1, 0], [4 * x * (y - 1 / 2), 1 / 2 * (4 * x**2 + 1)]])


def squeeze_hyper_strain(x: float, y: float) -> Tensor3D:
    """Hyper strain of the horizontal squeeze."""
    return Tensor3D([[[0, 0], [0, 0]], [[4 * (y - 1 / 2), 4 * x], [4 * x, 0]]])


deformations = [affine_deformation, bend_deformation, squeeze_deformation]
strains = [affine_strain, bend_strain, squeeze_strain]
hyper_strains = [affine_hyper_strain, bend_hyper_strain, squeeze_hyper_strain]


def create_random_barycentric_coordinates(
    num_coordinates,
) -> list[tuple[float, float, float]]:
    """Determines random barycentric coordinates for each provided triangle."""
    rng = np.random.default_rng()
    res = []
    for _ in range(num_coordinates):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        res.append((u, v, 1 - u - v))
    return res


def create_evaluation_points(
    barycentric_coordinates: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    points: list[Vector],
) -> list[Vector]:
    """Create a list of cartesian evaluation points given their barycentric coordinates."""
    return [
        points[i1] * w1 + points[i2] * w2 + points[i3] * w3
        for ((i1, i2, i3), (w1, w2, w3)) in zip(triangles, barycentric_coordinates)
    ]


def test_p1_interpolation() -> None:
    """Test piecewise affine interpolation."""
    eps = 1e-6
    grad_eps = 1e-2
    grid = SquareEquilateralGrid(num_horizontal_points=200)
    barycentric_coordinates = create_random_barycentric_coordinates(len(grid.triangles))
    evaluation_points = create_evaluation_points(
        barycentric_coordinates, grid.triangles, grid.points
    )

    for f, grad_f in zip(functions, gradients):
        params = [f(p[0], p[1]) for p in grid.points]
        f_approx = P1Interpolation(grid, params)

        values = [f(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            f_approx(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
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
    """Test piecewise affine interpolation with values given as pyomo paramters."""
    grid = SquareEquilateralGrid(num_horizontal_points=50)
    barycentric_coordinates = create_random_barycentric_coordinates(len(grid.triangles))

    for f in functions:
        params = [f(p[0], p[1]) for p in grid.points]
        f_approx = P1Interpolation(grid, params)
        values = [
            f_approx(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
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
            pyo.value(f_approx_pyomo(triangle, w))
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        assert values == approx(values_pyomo)

        grad_values = [f_approx.gradient(triangle).data for triangle in grid.triangles]
        grad_values_pyomo = [
            f_approx_pyomo.gradient(triangle).map(pyo.value).data
            for triangle in grid.triangles
        ]
        assert grad_values == grad_values_pyomo


def test_p1_deformation() -> None:
    """Test P1 deformation."""
    eps = 1e-6
    grad_eps = 1e-3
    grid = SquareEquilateralGrid(num_horizontal_points=100)
    barycentric_coordinates = create_random_barycentric_coordinates(len(grid.triangles))
    evaluation_points = create_evaluation_points(
        barycentric_coordinates, grid.triangles, grid.points
    )

    for deform, strain in zip(deformations, strains):
        params_vectors = [deform(p[0], p[1]) for p in grid.points]
        params = (
            [v[0] for v in params_vectors],
            [v[1] for v in params_vectors],
        )
        deform_approx = P1Deformation(grid, *params)

        values = [deform(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            deform_approx(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
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
    """Test interpolation via Bell's finite elements."""
    eps = 1e-6
    grad_eps = 1e-4
    hessian_eps = 1e-2
    grid = SquareEquilateralGrid(num_horizontal_points=14)
    barycentric_coordinates = create_random_barycentric_coordinates(len(grid.triangles))
    evaluation_points = create_evaluation_points(
        barycentric_coordinates, grid.triangles, grid.points
    )

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
            f_approx(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_error = sum(
            (value - value_approx) ** 2
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        grad_values = [grad_f(p[0], p[1]) for p in evaluation_points]
        grad_values_approx = [
            f_approx.gradient(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_grad_error = sum(
            (grad_value - grad_value_approx).normsqr()
            for (grad_value, grad_value_approx) in zip(grad_values, grad_values_approx)
        ) / len(grid.triangles)
        assert mean_squared_grad_error < grad_eps

        hessian_values = [hessian_f(p[0], p[1]) for p in evaluation_points]
        hessian_values_approx = [
            f_approx.hessian(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_hessian_error = sum(
            (hessian_value - hessian_value_approx).normsqr()
            for (hessian_value, hessian_value_approx) in zip(
                hessian_values, hessian_values_approx
            )
        ) / len(grid.triangles)
        assert mean_squared_hessian_error < hessian_eps


def test_c1_deformation() -> None:
    """Test C1 deformation."""
    eps = 1e-6
    grid = SquareEquilateralGrid(num_horizontal_points=7)
    barycentric_coordinates = create_random_barycentric_coordinates(len(grid.triangles))
    evaluation_points = create_evaluation_points(
        barycentric_coordinates, grid.triangles, grid.points
    )

    for deform, strain, hyper_strain in zip(deformations, strains, hyper_strains):
        deforms_at_grid_points = [deform(p[0], p[1]) for p in grid.points]
        strains_at_grid_points = [strain(p[0], p[1]) for p in grid.points]
        hyper_strains_at_grid_points = [hyper_strain(p[0], p[1]) for p in grid.points]
        params1 = [
            [y[0], G[0, 0], G[0, 1], H[0, 0, 0], H[0, 0, 1], H[0, 1, 1]]
            for (y, G, H) in zip(
                deforms_at_grid_points,
                strains_at_grid_points,
                hyper_strains_at_grid_points,
            )
        ]
        params2 = [
            [y[1], G[1, 0], G[1, 1], H[1, 0, 0], H[1, 0, 1], H[1, 1, 1]]
            for (y, G, H) in zip(
                deforms_at_grid_points,
                strains_at_grid_points,
                hyper_strains_at_grid_points,
            )
        ]
        deform_approx = C1Deformation(grid, params1, params2)

        values = [deform(p[0], p[1]) for p in evaluation_points]
        values_approx = [
            deform_approx(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_error = sum(
            (value - value_approx).normsqr()
            for (value, value_approx) in zip(values, values_approx)
        ) / len(grid.triangles)
        assert mean_squared_error < eps

        strain_values = [strain(p[0], p[1]) for p in evaluation_points]
        strain_values_approx = [
            deform_approx.strain(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_strain_error = sum(
            (strain_value - strain_value_approx).normsqr()
            for (strain_value, strain_value_approx) in zip(
                strain_values, strain_values_approx
            )
        ) / len(grid.triangles)
        assert mean_squared_strain_error < eps

        hyper_strain_values = [hyper_strain(p[0], p[1]) for p in evaluation_points]
        hyper_strain_values_approx = [
            deform_approx.hyper_strain(triangle, w)
            for (triangle, w) in zip(grid.triangles, barycentric_coordinates)
        ]
        mean_squared_hyper_strain_error = sum(
            (hyper_strain_value - hyper_strain_value_approx).normsqr()
            for (hyper_strain_value, hyper_strain_value_approx) in zip(
                hyper_strain_values, hyper_strain_values_approx
            )
        ) / len(grid.triangles)
        assert mean_squared_hyper_strain_error < eps
