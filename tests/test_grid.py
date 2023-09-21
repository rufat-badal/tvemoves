from tvemoves_rufbad.grid import SquareEquilateralGrid
from pytest import approx


def test_square_equilateral_grid():
    n = 50
    num_vertices = n * n
    num_edges = 2 * n * (n - 1) + (n - 1) * (n - 1)
    num_triangles = 2 * (n - 1) * (n - 1)
    num_boundary_vertices = 4 * (n - 1)
    num_boundary_edges = 4 * (n - 1)
    num_boundary_vertices_per_side = n
    num_boundary_edges_per_side = n - 1

    G = SquareEquilateralGrid(n)
    assert G.vertices == list(range(num_vertices))
    assert all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in G.initial_positions)
    assert len(G.edges) == num_edges
    assert len(G.edges) == len(set(G.edges))
    assert all(v in G.vertices and w in G.vertices for (v, w) in G.edges)
    assert all(
        (
            G.initial_positions[v][0] == approx(G.initial_positions[w][0])
            and abs(G.initial_positions[v][1] - G.initial_positions[w][1])
            == approx(G.grid_spacing)
        )
        or (
            abs(G.initial_positions[v][0] - G.initial_positions[w][0])
            == approx(G.grid_spacing)
            and G.initial_positions[v][1] == approx(G.initial_positions[w][1])
        )
        or (
            abs(G.initial_positions[v][0] - G.initial_positions[w][0])
            == approx(G.grid_spacing)
            and abs(G.initial_positions[v][1] - G.initial_positions[w][1])
            == approx(G.grid_spacing)
        )
        for (v, w) in G.edges
    )
    assert len(G.triangles) == num_triangles
    assert len(G.triangles) == len(set(G.triangles))
    assert len(G.boundary_vertices) == num_boundary_vertices
    assert all(
        G.initial_positions[v][0] == approx(0)
        or G.initial_positions[v][0] == approx(1)
        or G.initial_positions[v][1] == approx(0)
        or G.initial_positions[v][1] == approx(1)
        for v in G.boundary_vertices
    )
    assert len(G.boundary_edges) == num_boundary_edges
    assert all(
        v in G.boundary_vertices and w in G.boundary_vertices
        for (v, w) in G.boundary_edges
    )
    assert all(
        (
            G.initial_positions[v][0] == approx(G.initial_positions[w][0])
            and abs(G.initial_positions[v][1] - G.initial_positions[w][1])
            == approx(G.grid_spacing)
        )
        or (
            abs(G.initial_positions[v][0] - G.initial_positions[w][0])
            == approx(G.grid_spacing)
            and G.initial_positions[v][1] == approx(G.initial_positions[w][1])
        )
        for (v, w) in G.boundary_edges
    )
    assert all(e in G.edges for e in G.boundary_edges)
    assert G.neumann_vertices == G.boundary_vertices
    assert G.neumann_edges == G.boundary_edges
    assert G.dirichlet_vertices == []
    assert G.dirichlet_edges == []

    G_lower_fixed = SquareEquilateralGrid(n, fix="lower")
    assert all(
        G_lower_fixed.initial_positions[v][1] == approx(0)
        for v in G_lower_fixed.dirichlet_vertices
    )
    G_right_fixed = SquareEquilateralGrid(n, fix="right")
    assert all(
        G_right_fixed.initial_positions[v][0] == approx(1)
        for v in G_right_fixed.dirichlet_vertices
    )
    G_upper_fixed = SquareEquilateralGrid(n, fix="upper")
    assert all(
        G_upper_fixed.initial_positions[v][1] == approx(1)
        for v in G_upper_fixed.dirichlet_vertices
    )
    G_left_fixed = SquareEquilateralGrid(n, fix="left")
    assert all(
        G_left_fixed.initial_positions[v][0] == approx(0)
        for v in G_left_fixed.dirichlet_vertices
    )
