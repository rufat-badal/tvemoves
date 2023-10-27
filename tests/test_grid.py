"""Test grid classes."""

from pytest import approx
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.grid import (
    SquareEquilateralGrid,
    BarycentricPoint,
    BarycentricCurve,
    Curve,
    Grid,
)

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
    assert len(grid.boundary.vertices) == num_boundary_vertices
    assert all(
        grid.points[v][0] == approx(0)
        or grid.points[v][0] == approx(1)
        or grid.points[v][1] == approx(0)
        or grid.points[v][1] == approx(1)
        for v in grid.boundary.vertices
    )
    assert len(grid.boundary.edges) == num_boundary_edges
    assert all(
        v in grid.boundary.vertices and w in grid.boundary.vertices
        for (v, w) in grid.boundary.edges
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
        for (v, w) in grid.boundary.edges
    )
    assert all(e in grid.edges for e in grid.boundary.edges)
    assert grid.neumann_boundary.vertices == grid.boundary.vertices
    assert grid.neumann_boundary.edges == grid.boundary.edges
    assert grid.dirichlet_boundary.vertices == []
    assert grid.dirichlet_boundary.edges == []

    grid_lower_fixed = SquareEquilateralGrid(n, fix="lower")
    assert all(
        grid_lower_fixed.points[v][1] == approx(0)
        for v in grid_lower_fixed.dirichlet_boundary.vertices
    )
    grid_right_fixed = SquareEquilateralGrid(n, fix="right")
    assert all(
        grid_right_fixed.points[v][0] == approx(1)
        for v in grid_right_fixed.dirichlet_boundary.vertices
    )
    grid_upper_fixed = SquareEquilateralGrid(n, fix="upper")
    assert all(
        grid_upper_fixed.points[v][1] == approx(1)
        for v in grid_upper_fixed.dirichlet_boundary.vertices
    )
    grid_left_fixed = SquareEquilateralGrid(n, fix="left")
    assert all(
        grid_left_fixed.points[v][0] == approx(0)
        for v in grid_left_fixed.dirichlet_boundary.vertices
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


def test_generate_deformation_curve() -> None:
    """Test transformation from cartesian to barycentric curves."""
    grid = SquareEquilateralGrid(30)
    num_curves_horizontal = 3
    num_curves = 4 + 2 * num_curves_horizontal
    num_points_per_curve = 20
    deformation_curves = grid.generate_deformation_curves(
        num_points_per_curve, num_curves_horizontal
    )
    assert len(deformation_curves) == num_curves
    assert all(len(curve) == num_points_per_curve for curve in deformation_curves)


def to_cartesian_point(grid: Grid, p: BarycentricPoint) -> Vector:
    """Transform a barycentric point into a cartesian point."""
    triangle, barycentric_coordinates = p
    p1, p2, p3 = (grid.points[i] for i in triangle)
    w1, w2, w3 = barycentric_coordinates
    return w1 * p1 + w2 * p2 + w3 * p3


def to_cartesian_curve(grid: Grid, barycentric_curve: BarycentricCurve) -> Curve:
    """Transform a curve of barycentric points into the corresponding curve of cartesian points."""
    return [
        to_cartesian_point(grid, p_barycentric) for p_barycentric in barycentric_curve
    ]


def test_to_barycentric_curve() -> None:
    """Test transformation to a barycentric curve."""
    grid = SquareEquilateralGrid(30)
    num_curves_horizontal = 3
    num_points_per_curve = 20
    deformation_curves = grid.generate_deformation_curves(
        num_points_per_curve, num_curves_horizontal
    )
    error = 0.0
    for curve in deformation_curves:
        barycentric_curve = grid._to_barycentric_curve(curve)
        assert barycentric_curve is not None
        curve_recovered = to_cartesian_curve(grid, barycentric_curve)
        error += sum(
            (p - p_recovered).normsqr()
            for (p, p_recovered) in zip(curve, curve_recovered)
        ) / len(curve)
    assert error == approx(0)


def test_generate_barycentric_deformation_curves() -> None:
    """Test the generation of barycentric deformation curves."""
    grid = SquareEquilateralGrid(30)
    num_curves_horizontal = 3
    num_points_per_curve = 20
    deformation_curves = grid.generate_deformation_curves(
        num_points_per_curve, num_curves_horizontal
    )
    barycentric_deformation_curves = grid.generate_barycentric_deformation_curves(
        num_points_per_curve, num_curves_horizontal
    )
    error = 0.0
    for curve, barycentric_curve in zip(
        deformation_curves, barycentric_deformation_curves
    ):
        curve_recovered = to_cartesian_curve(grid, barycentric_curve)
        error += sum(
            (p - p_recovered).normsqr()
            for (p, p_recovered) in zip(curve, curve_recovered)
        ) / len(curve)
    assert error == approx(0)
