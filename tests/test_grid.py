"""Test grid classes."""

from pytest import approx
from tvemoves_rufbad.grid import SquareEquilateralGrid
from tests.test_interpolation import generate_random_barycentric_coordinates


def test_square_equilateral_grid() -> None:
    """Test square equilateral grid"""
    n = 50
    num_vertices = n * n
    num_edges = 2 * n * (n - 1) + (n - 1) * (n - 1)
    num_triangles = 2 * (n - 1) * (n - 1)
    num_boundary_vertices = 4 * (n - 1)
    num_boundary_edges = 4 * (n - 1)

    grid = SquareEquilateralGrid(n)
    grid_spacing = 1 / (n - 1)
    assert grid.vertices == list(range(num_vertices))
    assert all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in grid.points)
    assert len(grid.edges) == num_edges
    assert len(grid.edges) == len(set(grid.edges))
    assert all(v in grid.vertices and w in grid.vertices for (v, w) in grid.edges)
    assert all(
        (
            grid.points[v][0] == approx(grid.points[w][0])
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(grid_spacing)
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(grid_spacing)
            and grid.points[v][1] == approx(grid.points[w][1])
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(grid_spacing)
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(grid_spacing)
        )
        for (v, w) in grid.edges
    )
    assert len(grid.triangles) == num_triangles
    assert len(grid.triangles) == len(set(grid.triangles))
    assert len(grid.boundary_vertices) == num_boundary_vertices
    assert all(
        grid.points[v][0] == approx(0)
        or grid.points[v][0] == approx(1)
        or grid.points[v][1] == approx(0)
        or grid.points[v][1] == approx(1)
        for v in grid.boundary_vertices
    )
    assert len(grid.boundary_edges) == num_boundary_edges
    assert all(
        v in grid.boundary_vertices and w in grid.boundary_vertices
        for (v, w) in grid.boundary_edges
    )
    assert all(
        (
            grid.points[v][0] == approx(grid.points[w][0])
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(grid_spacing)
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(grid_spacing)
            and grid.points[v][1] == approx(grid.points[w][1])
        )
        for (v, w) in grid.boundary_edges
    )
    assert all(e in grid.edges for e in grid.boundary_edges)
    assert grid.neumann_vertices == grid.boundary_vertices
    assert grid.neumann_edges == grid.boundary_edges
    assert grid.dirichlet_vertices == []
    assert grid.dirichlet_edges == []

    grid_lower_fixed = SquareEquilateralGrid(n, fix="lower")
    assert all(
        grid_lower_fixed.points[v][1] == approx(0)
        for v in grid_lower_fixed.dirichlet_vertices
    )
    grid_right_fixed = SquareEquilateralGrid(n, fix="right")
    assert all(
        grid_right_fixed.points[v][0] == approx(1)
        for v in grid_right_fixed.dirichlet_vertices
    )
    grid_upper_fixed = SquareEquilateralGrid(n, fix="upper")
    assert all(
        grid_upper_fixed.points[v][1] == approx(1)
        for v in grid_upper_fixed.dirichlet_vertices
    )
    grid_left_fixed = SquareEquilateralGrid(n, fix="left")
    assert all(
        grid_left_fixed.points[v][0] == approx(0)
        for v in grid_left_fixed.dirichlet_vertices
    )


def test_area_coordinates() -> None:
    """Test transformation from barycentric to area coordinates."""
    grid = SquareEquilateralGrid(num_horizontal_points=50)
    barycentric_coordinates = generate_random_barycentric_coordinates(
        len(grid.triangles)
    )
    area_coordinates = [
        grid._area_coordinates(w, triangle)
        for (w, triangle) in zip(barycentric_coordinates, grid.triangles)
    ]
    assert all(sum(a) == approx(1) for a in area_coordinates)
