from tvemoves_rufbad.tensors import Vector, Matrix
from dataclasses import dataclass
from tvemoves_rufbad.bell_finite_element import (
    shape_function,
    shape_function_jacobian,
    shape_function_hessian_vectorized,
)


Edge = tuple[int, int]
Triangle = tuple[int, int, int]
BarycentricCoordinates = tuple[float, float, float]


@dataclass(frozen=True)
class Grid:
    vertices: list[int]
    edges: list[Edge]
    triangles: list[Triangle]
    boundary_vertices: list[int]
    boundary_edges: list[Edge]
    dirichlet_vertices: list[int]
    dirichlet_edges: list[Edge]
    neumann_vertices: list[int]
    neumann_edges: list[Edge]
    points: list[Vector]

    def _triangle_parameters(
        self, triangle: Triangle
    ) -> tuple[list[float], list[float], list[float], float,]:
        i1, i2, i3 = triangle
        z1, z2, z3 = (
            self.points[i1],
            self.points[i2],
            self.points[i3],
        )
        a = [
            z2[0] * z3[1] - z3[0] * z2[1],
            z3[0] * z1[1] - z1[0] * z3[1],
            z1[0] * z2[1] - z2[0] * z1[1],
        ]
        b = [z2[1] - z3[1], z3[1] - z1[1], z1[1] - z2[1]]
        c = [z3[0] - z2[0], z1[0] - z3[0], z2[0] - z1[0]]
        delta = (a[0] + a[1] + a[2]) / 2

        return a, b, c, delta

    def _area_coordinates(
        self, barycentric_coordinates: BarycentricCoordinates, triangle: Triangle
    ) -> list[float]:
        a, b, c, delta = self._triangle_parameters(triangle)
        triangle_vertices = [self.points[i] for i in triangle]
        # barycentric coordinates => cartesian coordinates
        x: float = sum(
            wi * zi[0] for (wi, zi) in zip(barycentric_coordinates, triangle_vertices)
        )
        y: float = sum(
            wi * zi[1] for (wi, zi) in zip(barycentric_coordinates, triangle_vertices)
        )

        return [(ai + bi * x + ci * y) / (2 * delta) for (ai, bi, ci) in zip(a, b, c)]

    def gradient_transform(
        self,
        triangle: Triangle,
        area_gradient: Vector,
    ) -> Vector:
        _, b, c, delta = self._triangle_parameters(triangle)
        trafo_matrix = Matrix([[b[0], b[1], b[2]], [c[0], c[1], c[2]]]) / (2 * delta)
        return trafo_matrix.dot(area_gradient)

    def hessian_transform(
        self,
        triangle: Triangle,
        area_hessian_vectorized: Vector,
    ) -> Matrix:
        _, b, c, delta = self._triangle_parameters(triangle)
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
        flat_hessian = trafo_matrix.dot(area_hessian_vectorized)
        return Matrix(
            [[flat_hessian[0], flat_hessian[2]], [flat_hessian[2], flat_hessian[1]]]
        )

    def shape_function(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Vector:
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function(
            *self._area_coordinates(barycentric_coordinates, triangle), *b, *c
        )

    def shape_function_jacobian(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Matrix:
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_jacobian(
            *self._area_coordinates(barycentric_coordinates, triangle), *b, *c
        )

    def shape_function_hessian_vectorized(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Matrix:
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_hessian_vectorized(
            *self._area_coordinates(barycentric_coordinates, triangle), *b, *c
        )


def generate_square_equilateral_grid(
    num_horizontal_points: int, fix: str = "none"
) -> Grid:
    def pair_to_vertex(i, j):
        # (0, 0) -> 0, (1, 0) -> 1, ...,
        # (num_horizontal_points - 1, 0) -> num_horizontal_points -1,
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

    grid_spacing = 1 / (num_horizontal_points - 1)
    points = [
        Vector(
            [
                v % num_horizontal_points * grid_spacing,
                v // num_horizontal_points * grid_spacing,
            ]
        )
        for v in vertices
    ]

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
        points,
    )
