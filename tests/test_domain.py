"""Test domains."""

from pytest import approx
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.domain import RectangleDomain, Grid, BarycentricPoint


def test_rectangle_domain_grid() -> None:
    """Test grid factory of a rectangle domain"""
    width = 3
    height = 2
    # make sure that scale does not evenly divide width and height
    scale = 0.7 / (width * height)
    num_vertical_vertices = int(height / scale) + 1
    num_horizontal_vertices = int(width / scale) + 1
    grid_width = (num_horizontal_vertices - 1) * scale
    grid_height = (num_vertical_vertices - 1) * scale
    num_vertices = num_vertical_vertices * num_horizontal_vertices
    num_vertical_edges = num_horizontal_vertices * (num_vertical_vertices - 1)
    num_horizontal_edges = num_vertical_vertices * (num_horizontal_vertices - 1)
    num_diagonal_edges = (num_vertical_vertices - 1) * (num_horizontal_vertices - 1)
    num_edges = num_vertical_edges + num_horizontal_edges + num_diagonal_edges
    num_triangles = 2 * num_diagonal_edges
    num_boundary_vertices = 2 * (num_vertical_vertices - 1) + 2 * (num_horizontal_vertices - 1)
    num_boundary_edges = num_boundary_vertices

    rectangle = RectangleDomain(width, height)
    grid = rectangle.grid(scale)
    assert grid.vertices == list(range(num_vertices))
    assert all(0 <= p[0] <= width and 0 <= p[1] <= height for p in grid.points)
    assert len(grid.edges) == num_edges
    assert len(grid.edges) == len(set(grid.edges))
    assert all(v in grid.vertices and w in grid.vertices for (v, w) in grid.edges)
    assert all(
        (
            grid.points[v][0] == approx(grid.points[w][0])
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(scale)
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(scale)
            and grid.points[v][1] == approx(grid.points[w][1])
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(scale)
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(scale)
        )
        for (v, w) in grid.edges
    )
    assert len(grid.triangles) == num_triangles
    assert len(grid.triangles) == len(set(grid.triangles))
    assert len(grid.boundary.vertices) == num_boundary_vertices
    assert all(
        grid.points[v][0] == approx(0)
        or grid.points[v][0] == approx(grid_width)
        or grid.points[v][1] == approx(0)
        or grid.points[v][1] == approx(grid_height)
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
            and abs(grid.points[v][1] - grid.points[w][1]) == approx(scale)
        )
        or (
            abs(grid.points[v][0] - grid.points[w][0]) == approx(scale)
            and grid.points[v][1] == approx(grid.points[w][1])
        )
        for (v, w) in grid.boundary.edges
    )
    assert all(e in grid.edges for e in grid.boundary.edges)
    assert grid.neumann_boundary.vertices == grid.boundary.vertices
    assert grid.neumann_boundary.edges == grid.boundary.edges
    assert not grid.dirichlet_boundary.vertices
    assert not grid.dirichlet_boundary.edges

    rectangle_lower_fixed = RectangleDomain(width, height, fix="lower")
    grid_lower_fixed = rectangle_lower_fixed.grid(scale)
    assert all(
        grid_lower_fixed.points[v][1] == approx(0)
        for v in grid_lower_fixed.dirichlet_boundary.vertices
    )

    rectangle_right_fixed = RectangleDomain(width, height, fix="right")
    grid_right_fixed = rectangle_right_fixed.grid(scale)
    assert all(
        grid_right_fixed.points[v][0] == approx(grid_width)
        for v in grid_right_fixed.dirichlet_boundary.vertices
    )

    rectangle_upper_fixed = RectangleDomain(width, height, fix="upper")
    grid_upper_fixed = rectangle_upper_fixed.grid(scale)
    assert all(
        grid_upper_fixed.points[v][1] == approx(grid_height)
        for v in grid_upper_fixed.dirichlet_boundary.vertices
    )

    rectangle_left_fixed = RectangleDomain(width, height, fix="left")
    grid_left_fixed = rectangle_left_fixed.grid(scale)
    assert all(
        grid_left_fixed.points[v][0] == approx(0)
        for v in grid_left_fixed.dirichlet_boundary.vertices
    )


def test_rectangle_domain_curves() -> None:
    """Test curves of rectangle domains."""
    width = 3
    height = 2
    num_points_per_curve = 100
    num_horizontal_curves = 3
    num_vertical_curves = 5

    rectangle = RectangleDomain(width, height)
    curves = rectangle.curves(num_points_per_curve, num_horizontal_curves, num_vertical_curves)
    horizontal_step = [width / (num_points_per_curve - 1), 0]
    vertical_step = [0, height / (num_points_per_curve - 1)]

    for curve in curves:
        assert len(curve) == num_points_per_curve
        first_step = (curve[1] - curve[0]).data
        assert first_step == approx(horizontal_step) or first_step == approx(vertical_step)
        assert all(
            (curve[i + 1] - curve[i]).data == approx(first_step)
            for i in range(2, num_points_per_curve - 1)
        )
        assert curve[0][0] == approx(0) or curve[0][1] == approx(0)
        assert curve[-1][0] == approx(width) or curve[-1][1] == approx(height)


def _to_cartesian_point(barycentric_point: BarycentricPoint, grid: Grid) -> Vector:
    i1, i2, i3 = barycentric_point.triangle
    p1, p2, p3 = grid.points[i1], grid.points[i2], grid.points[i3]
    coords = barycentric_point.coordinates
    return coords.u * p1 + coords.v * p2 + coords.w * p3


def test_to_barycentric_curve() -> None:
    """Test transformation from a curve to a barycentric curve."""
    width = 1
    height = 1
    num_points_per_curve = 2
    num_horizontal_curves = 0
    num_vertical_curves = 0

    rectangle = RectangleDomain(width, height)
    curves = rectangle.curves(num_points_per_curve, num_horizontal_curves, num_vertical_curves)
    scale = 1 / 5
    grid = rectangle.grid(scale)
    barycentric_curves = [grid.to_barycentric_curve(curve) for curve in curves]
    for barycentric_curve, curve in zip(barycentric_curves, curves):
        assert len(barycentric_curve) == len(curve)
        if len(barycentric_curve) < len(curve):
            for barycentric_point, point in zip(barycentric_curve, curve):
                assert (_to_cartesian_point(barycentric_point, grid) - point).norm() == approx(0)
