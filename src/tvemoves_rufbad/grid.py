from dataclasses import dataclass


@dataclass
class Grid:
    vertices: list
    edges: list
    triangles: list
    boundary_vertices: list
    boundary_edges: list
    dirichlet_vertices: list
    dirichlet_edges: list
    neumann_vertices: list
    neumann_edges: list


def generate_square_grid(num_horizontal_points, fix=None):
    vertices = [
        (i, j)
        for j in range(num_horizontal_points)
        for i in range(num_horizontal_points)
    ]

    horizontal_edges = [
        ((i, j), (i + 1, j))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points)
    ]
    vertical_edges = [
        ((i, j), (i, j + 1))
        for i in range(num_horizontal_points)
        for j in range(num_horizontal_points - 1)
    ]
    diagonal_edges = [
        ((i, j), (i + 1, j + 1))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    edges = horizontal_edges + vertical_edges + diagonal_edges

    lower_triangles = [
        ((i, j), (i + 1, j), (i + 1, j + 1))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    upper_triangles = [
        ((i, j), (i, j + 1), (i + 1, j + 1))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    triangles = lower_triangles + upper_triangles

    lower_boundary_vertices = set((i, 0) for i in range(num_horizontal_points))
    lower_boundary_edges = [
        ((i, 0), (i + 1, 0)) for i in range(num_horizontal_points - 1)
    ]
    right_boundary_vertices = set(
        (num_horizontal_points - 1, i) for i in range(num_horizontal_points)
    )
    right_boundary_edges = [
        ((num_horizontal_points - 1, i), (num_horizontal_points - 1, i + 1))
        for i in range(num_horizontal_points - 1)
    ]
    upper_boundary_vertices = set(
        (i, num_horizontal_points - 1) for i in range(num_horizontal_points)
    )
    upper_boundary_edges = [
        ((i, num_horizontal_points - 1), (i + 1, num_horizontal_points - 1))
        for i in range(num_horizontal_points - 1)
    ]
    left_boundary_vertices = set((0, i) for i in range(num_horizontal_points))
    left_boundary_edges = [
        ((0, i), (0, i + 1)) for i in range(num_horizontal_points - 1)
    ]

    boundary_vertices = list(
        lower_boundary_vertices.union(right_boundary_vertices)
        .union(upper_boundary_vertices)
        .union(left_boundary_vertices)
    )
    boundary_edges = (
        lower_boundary_edges
        + right_boundary_edges
        + upper_boundary_edges
        + left_boundary_edges
    )

    match fix:
        case None:
            dirichlet_vertices = []
            dirichlet_edges = []
            neumann_vertices = boundary_vertices
            neumann_edges = boundary_edges
        case "all": 
            dirichlet_vertices = boundary_vertices
            dirichlet_edges = boundary_edges
            neumann_vertices = []
            neumann_edges = []
        case "lower":
            dirichlet_vertices = list(lower_boundary_vertices)
            dirichlet_edges = lower_boundary_edges
            neumann_vertices = list(
                right_boundary_vertices.union(upper_boundary_vertices).union(
                    left_boundary_vertices
                )
            )
            neumann_edges = (
                right_boundary_edges + upper_boundary_edges + left_boundary_edges
            )
        case "right":
            dirichlet_vertices = list(right_boundary_vertices)
            dirichlet_edges = right_boundary_edges
            neumann_vertices = list(
                upper_boundary_vertices.union(left_boundary_vertices).union(
                    lower_boundary_vertices
                )
            )
            neumann_edges = (
                upper_boundary_edges + left_boundary_edges + lower_boundary_edges
            )
        case "upper":
            dirichlet_vertices = list(upper_boundary_vertices)
            dirichlet_edges = upper_boundary_edges
            neumann_vertices = list(
                left_boundary_vertices.union(lower_boundary_vertices).union(
                    right_boundary_vertices
                )
            )
            neumann_edges = (
                left_boundary_edges + lower_boundary_edges + right_boundary_edges
            )
        case "left":
            dirichlet_vertices = list(left_boundary_vertices)
            dirichlet_edges = left_boundary_edges
            neumann_vertices = list(
                lower_boundary_vertices.union(right_boundary_vertices).union(
                    upper_boundary_vertices
                )
            )
            neumann_edges = (
                lower_boundary_edges + right_boundary_edges + upper_boundary_edges
            )
        case _:
            raise ValueError("invalid dirichlet condition provided")

    return Grid(
        vertices,
        edges,
        triangles,
        boundary_vertices,
        boundary_edges,
        dirichlet_vertices,
        dirichlet_edges,
        neumann_vertices,
        neumann_edges,
    )


grid = generate_square_grid(3)
print(grid)
