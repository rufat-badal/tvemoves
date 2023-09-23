from .tensors import Vector, Matrix


class SquareEquilateralGrid:
    def __init__(self, num_horizontal_points, fix=None):
        def pair_to_vertex(i, j):
            # (0, 0) -> 0, (1, 0) -> 1, ..., (num_horizontal_points - 1, 0) -> num_horizontal_points -1,
            # (0, 1) -> num_horizontal_points, ...
            return j * num_horizontal_points + i

        self.vertices = list(range(num_horizontal_points * num_horizontal_points))

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
        self.edges = horizontal_edges + vertical_edges + diagonal_edges

        lower_triangles = [
            (
                pair_to_vertex(i + 1, j),
                pair_to_vertex(i, j),
                pair_to_vertex(i + 1, j + 1),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points - 1)
        ]
        upper_triangles = [
            (
                pair_to_vertex(i, j + 1),
                pair_to_vertex(i + 1, j + 1),
                pair_to_vertex(i, j),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_horizontal_points - 1)
        ]
        self.triangles = lower_triangles + upper_triangles

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

        self.boundary_vertices = list(
            lower_boundary_vertices.union(right_boundary_vertices)
            .union(upper_boundary_vertices)
            .union(left_boundary_vertices)
        )
        self.boundary_edges = (
            lower_boundary_edges
            + right_boundary_edges
            + upper_boundary_edges
            + left_boundary_edges
        )

        match fix:
            case None:
                self.dirichlet_vertices = []
                self.dirichlet_edges = []
                self.neumann_vertices = self.boundary_vertices
                self.neumann_edges = self.boundary_edges
            case "all":
                self.dirichlet_vertices = self.boundary_vertices
                self.dirichlet_edges = self.boundary_edges
                self.neumann_vertices = []
                self.neumann_edges = []
            case "lower":
                self.dirichlet_vertices = list(lower_boundary_vertices)
                self.dirichlet_edges = lower_boundary_edges
                self.neumann_vertices = list(
                    right_boundary_vertices.union(upper_boundary_vertices).union(
                        left_boundary_vertices
                    )
                )
                self.neumann_edges = (
                    right_boundary_edges + upper_boundary_edges + left_boundary_edges
                )
            case "right":
                self.dirichlet_vertices = list(right_boundary_vertices)
                self.dirichlet_edges = right_boundary_edges
                self.neumann_vertices = list(
                    upper_boundary_vertices.union(left_boundary_vertices).union(
                        lower_boundary_vertices
                    )
                )
                self.neumann_edges = (
                    upper_boundary_edges + left_boundary_edges + lower_boundary_edges
                )
            case "upper":
                self.dirichlet_vertices = list(upper_boundary_vertices)
                self.dirichlet_edges = upper_boundary_edges
                self.neumann_vertices = list(
                    left_boundary_vertices.union(lower_boundary_vertices).union(
                        right_boundary_vertices
                    )
                )
                self.neumann_edges = (
                    left_boundary_edges + lower_boundary_edges + right_boundary_edges
                )
            case "left":
                self.dirichlet_vertices = list(left_boundary_vertices)
                self.dirichlet_edges = left_boundary_edges
                self.neumann_vertices = list(
                    lower_boundary_vertices.union(right_boundary_vertices).union(
                        upper_boundary_vertices
                    )
                )
                self.neumann_edges = (
                    lower_boundary_edges + right_boundary_edges + upper_boundary_edges
                )
            case _:
                raise ValueError("invalid dirichlet condition provided")

        self.grid_spacing = 1 / (num_horizontal_points - 1)
        self.points = [
            Vector(
                [
                    v % num_horizontal_points * self.grid_spacing,
                    v // num_horizontal_points * self.grid_spacing,
                ]
            )
            for v in self.vertices
        ]

    def _generate_triangle_parameters(self, triangle):
        i, j, k = triangle
        x, y, z = (
            self.points[i],
            self.points[j],
            self.points[k],
        )
        a = [
            y[0] * z[1] - z[0] * y[1],
            z[0] * x[1] - x[0] * z[1],
            x[0] * y[1] - y[0] * x[1],
        ]
        b = [y[1] - z[1], z[1] - x[1], x[1] - y[1]]
        c = [z[0] - y[0], x[0] - z[0], y[0] - x[0]]
        delta = (a[0] + a[1] + a[2]) / 2

        return b, c, delta

    def gradient_transform(self, triangle, barycentric_gradient):
        b, c, delta = self._generate_triangle_parameters(triangle)
        trafo_matrix = Matrix([[b[0], b[1], b[2]], [c[0], c[1], c[2]]]) / (2 * delta)
        return trafo_matrix.dot(barycentric_gradient)

    def hessian_transform(self, triangle, barycentric_hessian):
        b, c, delta = self._generate_triangle_parameters(triangle)
        trafo_matrix = Matrix(
            [
                [
                    b[0] ** 2,
                    b[1] ** 2,
                    b[2] ** 2,
                    2 * b[0] * b[1],
                    2 * b[0] * b[2],
                    2 * b[1] * b[2],
                ],
                [
                    c[0] ** 2,
                    c[1] ** 2,
                    c[2] ** 2,
                    2 * c[0] * c[1],
                    2 * c[0] * c[2],
                    2 * c[1] * c[2],
                ],
                [
                    b[0] * c[0],
                    b[1] * c[1],
                    b[2] * c[2],
                    b[0] * c[1] + b[1] * c[0],
                    b[0] * c[2] + b[2] * c[0],
                    b[1] * c[2] + b[2] * c[1],
                ],
            ]
        ) / (4 * delta**2)
        flat_hessian = trafo_matrix.dot(barycentric_hessian.flatten())
        return Matrix(
            [[flat_hessian[0], flat_hessian[2]], [flat_hessian[2], flat_hessian[1]]]
        )
