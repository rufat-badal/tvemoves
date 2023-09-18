from tvemoves_rufbad.grid import generate_square_grid


def test_generate_square_grid():
    n = 50
    num_vertices = n * n
    num_edges = 2 * n * (n - 1) + (n - 1) * (n - 1)
    num_triangles = 2 * (n - 1) * (n - 1)
    num_boundary_vertices = 4 * (n - 1)
    num_boundary_edges = 4 * (n - 1)

    G = generate_square_grid(n)
    assert len(G.vertices) == num_vertices
    assert len(G.vertices) == len(set(G.vertices))
    assert len(G.edges) == num_edges
    assert len(G.edges) == len(set(G.edges))
    assert all(v in G.vertices and w in G.vertices for (v, w) in G.edges)
    assert all(abs(v[0] - w[0]) <= 1 and abs(v[1] - w[1]) <= 1 for (v, w) in G.edges)
    assert len(G.triangles) == num_triangles
    assert len(G.triangles) == len(set(G.triangles))
    assert all((i, j) in G.vertices for i in range(n) for j in range(n))
    assert len(G.boundary_vertices) == num_boundary_vertices
    assert all(0 <= i < n and 0 <= j < n for (i, j) in G.boundary_vertices)
    assert all(i in [0, n - 1] or j in [0, n - 1] for (i, j) in G.boundary_vertices)
    assert len(G.boundary_edges) == num_boundary_edges
    assert all(
        v in G.boundary_vertices and w in G.boundary_vertices
        for (v, w) in G.boundary_edges
    )
    assert all(e in G.edges for e in G.boundary_edges)
