from tvemoves_rufbad.grid import SquareEquilateralGrid


def test_square_equilateral_grid():
    n = 50
    num_vertices = n * n
    num_edges = 2 * n * (n - 1) + (n - 1) * (n - 1)
    num_triangles = 2 * (n - 1) * (n - 1)
    num_boundary_vertices = 4 * (n - 1)
    num_boundary_edges = 4 * (n - 1)

    G = SquareEquilateralGrid(n)
    assert G.vertices == list(range(num_vertices))
    assert len(G.edges) == num_edges
    assert len(G.edges) == len(set(G.edges))
    assert all(v in G.vertices and w in G.vertices for (v, w) in G.edges)
    assert len(G.triangles) == num_triangles
    assert len(G.triangles) == len(set(G.triangles))
    assert len(G.boundary_vertices) == num_boundary_vertices
    assert len(G.boundary_edges) == num_boundary_edges
    assert all(
        v in G.boundary_vertices and w in G.boundary_vertices
        for (v, w) in G.boundary_edges
    )
    assert all(e in G.edges for e in G.boundary_edges)
