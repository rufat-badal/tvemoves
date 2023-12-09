"""Test all interpolations"""

from tvemoves_rufbad.interpolation import P1Interpolation, C1Interpolation
from tvemoves_rufbad.domain import RectangleDomain, Triangle, Grid
from tvemoves_rufbad.tensors import Vector, BarycentricCoordinates
from .helpers import random_polynomial_2d, random_barycentric_coordinates, point_c1_params

GRID_WIDTH = 3
GRID_HEIGHT = 4
GRID_SCALE = 0.5
EPS = 1e-10


def _random_test_points(
    grid: Grid,
    points_per_triangle: int = 1,
) -> list[tuple[tuple[Triangle, BarycentricCoordinates], Vector]]:
    test_points = []
    for triangle in grid.triangles:
        p1, p2, p3 = grid.triangle_vertices(triangle)
        for bc in random_barycentric_coordinates(points_per_triangle):
            test_points.append(((triangle, bc), bc.l1 * p1 + bc.l2 * p2 + bc.l3 * p3))

    return test_points


def test_p1_interpolation() -> None:
    """Test piecewise affine interpolation"""
    poly, poly_gradient = random_polynomial_2d(degree=1, num_derivatives=1)
    rect = RectangleDomain(GRID_WIDTH, GRID_HEIGHT)
    grid = rect.grid(GRID_SCALE)

    params = [poly(*p) for p in grid.points]
    interpolation = P1Interpolation(grid, params)

    for (triangle, bc), ec in _random_test_points(grid):
        assert abs(poly(*ec) - interpolation(triangle, bc)) < EPS
        assert (poly_gradient(*ec) - interpolation.gradient(triangle, bc)).norm() < EPS


def test_c1_interpolation() -> None:
    """Test C1 interpolation"""
    poly, poly_gradient, poly_hessian = random_polynomial_2d(degree=4, num_derivatives=2)
    rect = RectangleDomain(GRID_WIDTH, GRID_HEIGHT)
    grid = rect.grid(GRID_SCALE)

    params = [point_c1_params(p, poly, poly_gradient, poly_hessian) for p in grid.points]
    interpolation = C1Interpolation(grid, params)
    for (triangle, bc), ec in _random_test_points(grid):
        assert abs(poly(*ec) - interpolation(triangle, bc)) < EPS
        assert (poly_gradient(*ec) - interpolation.gradient(triangle, bc)).norm() < EPS
        assert (poly_hessian(*ec) - interpolation.hessian(triangle, bc)).norm() < 10 * EPS
