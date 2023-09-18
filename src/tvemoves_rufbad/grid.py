class Grid:
    def __init__(
        self,
        vertices: list[int],
        edges: list[tuple[int, int]],
        triangles: list[tuple[int, int, int]],
        boundary_vertices: list[int],
        boundary_edges: list[tuple[int, int]],
        dirichlet_vertices: list[int],
        dirichlet_edges: list[tuple[int, int]],
        neumann_vertices: list[int],
        neumann_edges: list[tuple[int, int]],
    ):
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles
        self.boundary_vertices = boundary_vertices
        self.boundary_edges = boundary_edges
        self.dirichlet_vertices = dirichlet_vertices
        self.dirichlet_edges = dirichlet_edges
        self.neumann_vertices = neumann_vertices
        self.neumann_edges = neumann_edges


class SquareEquilateralGrid(Grid):
    def __init__(self, num_horizontal_points: int, fix: str = "none"):
        def pair_to_vertex(i: int, j: int):
            # (0, 0) -> 0, (1, 0) -> 1, ..., (num_horizontal_points - 1, 0) -> num_horizontal_points -1,
            # (0, 1) -> num_horizontal_points, ...
            return j * num_horizontal_points + i

        vertices = list(range(num_horizontal_points * num_horizontal_points))

        horizontal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j))
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points)
        ]
        vertical_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i, j + 1))
            for i in range(num_horizontal_points)
            for j in range(num_horizontal_points - 1)
        ]
        diagonal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j + 1))
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points - 1)
        ]
        edges = horizontal_edges + vertical_edges + diagonal_edges

        lower_triangles = [
            (
                pair_to_vertex(i, j),
                pair_to_vertex(i + 1, j),
                pair_to_vertex(i + 1, j + 1),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points - 1)
        ]
        upper_triangles = [
            (
                pair_to_vertex(i, j),
                pair_to_vertex(i, j + 1),
                pair_to_vertex(i + 1, j + 1),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points - 1)
        ]
        triangles = lower_triangles + upper_triangles

        lower_boundary_vertices = set(
            pair_to_vertex(i, 0) for i in range(num_horizontal_points)
        )
        lower_boundary_edges = [
            (pair_to_vertex(i, 0), pair_to_vertex(i + 1, 0))
            for i in range(num_horizontal_points - 1)
        ]
        right_boundary_vertices = set(
            pair_to_vertex(num_horizontal_points - 1, i)
            for i in range(num_horizontal_points)
        )
        right_boundary_edges = [
            (
                pair_to_vertex(num_horizontal_points - 1, i),
                pair_to_vertex(num_horizontal_points - 1, i + 1),
            )
            for i in range(num_horizontal_points - 1)
        ]
        upper_boundary_vertices = set(
            pair_to_vertex(i, num_horizontal_points - 1)
            for i in range(num_horizontal_points)
        )
        upper_boundary_edges = [
            (
                pair_to_vertex(i, num_horizontal_points - 1),
                pair_to_vertex(i + 1, num_horizontal_points - 1),
            )
            for i in range(num_horizontal_points - 1)
        ]
        left_boundary_vertices = set(
            pair_to_vertex(0, i) for i in range(num_horizontal_points)
        )
        left_boundary_edges = [
            (pair_to_vertex(0, i), pair_to_vertex(0, i + 1))
            for i in range(num_horizontal_points - 1)
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
            case "none":
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

        super().__init__(
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


G = SquareEquilateralGrid(10)
