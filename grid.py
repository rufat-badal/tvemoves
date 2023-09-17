from collections import namedtuple


class Grid(NamedTuple):
    vertices
    points
    edges
    triangles
    boudary_vertices
    boundary_edges
    dirichlet_vertices
    dirichlet_edges
    neumann_vertices
    neumann_edges


def square_grid_factory(num_horizontal_vertices, fix=None):
    pass
