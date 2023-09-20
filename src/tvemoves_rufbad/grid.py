from .tensors import Vector, Matrix


class Grid:
    def __init__(
        self,
        vertices,
        edges,
        triangles,
        boundary_vertices,
        boundary_edges,
        dirichlet_vertices,
        dirichlet_edges,
        neumann_vertices,
        neumann_edges,
        initial_points,
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
        self.initial_points = initial_points


class SquareEquilateralGrid(Grid):
    def __init__(self, num_horizontal_points, fix=None):
        def pair_to_vertex(i, j):
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

        grid_spacing = 1 / (num_horizontal_points - 1)
        initial_points = [
            Vector(
                [
                    v % num_horizontal_points * grid_spacing,
                    v // num_horizontal_points * grid_spacing,
                ]
            )
            for v in vertices
        ]

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
            initial_points,
        )

        self.grid_spacing = grid_spacing

    def _generate_triangle_parameters(self, triangle):
        i, j, k = triangle
        x, y, z = self.initial_points[i], self.initial_points[j], self.initial_points[k]
        a = [
            y[0] * z[1] - z[0] * y[1],
            z[0] * x[1] - x[0] * z[1],
            x[0] * y[1] - y[0] * x[1],
        ]
        b = [y[1] - z[1], z[1] - x[1], x[1] - y[1]]
        c = [z[0] - y[0], x[0] - z[0], y[0] - x[0]]
        delta = (a[1] + a[2] + a[3]) / 2

        return b, c, delta

    def gradient_transform(self, triangle, barycentric_gradient):
        b, c, delta = self._generate_triangle_parameters(triangle)
        trafo_matrix = Matrix([[b[1], b[2], b[3]], [c[1], c[2], c[3]]]) / (2 * delta)
        return trafo_matrix.dot(barycentric_gradient)

    def hessian_transform(self, triangle, barycentric_hessian):
        b, c, delta = self._generate_triangle_parameters(triangle)
        trafo_matrix = Matrix(
            [
                [
                    b[1] ** 2,
                    b[2] ** 2,
                    b[3] ** 2,
                    2 * b[1] * b[2],
                    2 * b[1] * b[3],
                    2 * b[2] * b[3],
                ],
                [
                    c[1] ** 2,
                    c[2] ** 2,
                    c[3] ** 2,
                    2 * c[1] * c[2],
                    2 * c[1] * c[3],
                    2 * c[2] * c[3],
                ],
                [
                    b[1] * c[1],
                    b[2] * c[2],
                    b[3] * c[3],
                    b[1] * c[2] + b[2] * c[1],
                    b[1] * c[3] + b[3] * c[1],
                    b[2] * c[3] + b[3] * c[2],
                ],
            ]
        ) / (4 * delta**2)
        flat_hessian = trafo_matrix.dot(barycentric_hessian.flatten())
        return Matrix(
            [[flat_hessian[1], flat_hessian[3]], [flat_hessian[3], flat_hessian[2]]]
        )
