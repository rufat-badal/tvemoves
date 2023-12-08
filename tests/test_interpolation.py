"""Test all interpolations"""

from tvemoves_rufbad.interpolation import P1Interpolation
from tvemoves_rufbad.domain import RectangleDomain
from .helpers import random_polynomial_2d, random_barycentric_coordinates

GRID_WIDTH = 3
GRID_HEIGHT = 4
GRID_SCALE = 0.5
EPS = 1e-13


def test_p1_interpolation() -> None:
    """Test piecewise affine interpolation"""
    poly, poly_gradient = random_polynomial_2d(1, 1)
    rect = RectangleDomain(GRID_WIDTH, GRID_HEIGHT)
    grid = rect.grid(GRID_SCALE)

    params = [poly(*p) for p in grid.points]
    interpolation = P1Interpolation(grid, params)
    # One barycentric coordinate per triangle
    barycentric_coordinates = random_barycentric_coordinates(len(grid.triangles))
    euclidean_coordinates = []
    for bc, triangle in zip(barycentric_coordinates, grid.triangles):
        p1, p2, p3 = grid.triangle_vertices(triangle)
        euclidean_coordinates.append(bc.u * p1 + bc.v * p2 + bc.w * p3)

    for bc, triangle, ec in zip(
        barycentric_coordinates, grid.triangles, euclidean_coordinates
    ):
        assert abs(poly(*ec) - interpolation(triangle, bc)) < EPS
        assert (poly_gradient(*ec) - interpolation.gradient(triangle)).norm() < EPS
