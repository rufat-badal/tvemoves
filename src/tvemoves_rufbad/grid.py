"""Module providing grid generators."""

from collections import defaultdict
from abc import ABC, abstractmethod
from tvemoves_rufbad.tensors import Vector, Matrix
from tvemoves_rufbad.bell_finite_element import (
    shape_function,
    shape_function_jacobian,
    shape_function_hessian_vectorized,
    shape_function_on_edge_left,
    shape_function_on_edge_right,
)


Edge = tuple[int, int]
Triangle = tuple[int, int, int]
BarycentricCoordinates = tuple[float, float, float]
BarycentricPoint = tuple[Triangle, BarycentricCoordinates]
BarycentricCurve = list[BarycentricPoint]
Curve = list[Vector]


class Grid(ABC):
    """Abstract grid class."""

    def __init__(
        self,
        vertices: list[int],
        edges: list[Edge],
        triangles: list[Triangle],
        boundary_vertices: list[int],
        boundary_edges: list[Edge],
        dirichlet_vertices: list[int],
        dirichlet_edges: list[Edge],
        neumann_vertices: list[int],
        neumann_edges: list[Edge],
        points: list[Vector],
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
        self.points = points

        self.opposite_vertices = defaultdict(list)
        for triangle in triangles:
            for i in range(3):
                edge = (triangle[i], triangle[(i + 1) % 3])
                edge_reverse = (edge[1], edge[0])
                opposite_vertex = triangle[(i + 2) % 3]
                self.opposite_vertices[edge].append(opposite_vertex)
                self.opposite_vertices[edge_reverse].append(opposite_vertex)

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
        p1, p2, p3 = (self.points[i] for i in triangle)
        w1, w2, w3 = barycentric_coordinates

        # barycentric coordinates => cartesian coordinates
        p = w1 * p1 + w2 * p2 + w3 * p3

        return [
            (ai + bi * p[0] + ci * p[1]) / (2 * delta) for (ai, bi, ci) in zip(a, b, c)
        ]

    def gradient_transform(
        self,
        triangle: Triangle,
        area_gradient: Vector,
    ) -> Vector:
        """Transforms gradient with respect to area coordinates to a Cartesian gradient"""
        _, b, c, delta = self._triangle_parameters(triangle)
        trafo_matrix = Matrix([[b[0], b[1], b[2]], [c[0], c[1], c[2]]]) / (2 * delta)
        return trafo_matrix.dot(area_gradient)

    def hessian_transform(
        self,
        triangle: Triangle,
        area_hessian_vectorized: Vector,
    ) -> Matrix:
        """Transforms vectorized hessian with respect to area coordinates to Cartesian hessian."""
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
        """Computes shape function in a triangle."""
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function(
            self._area_coordinates(barycentric_coordinates, triangle), b, c
        )

    def shape_function_jacobian(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Matrix:
        """Computes jacobian of the shape function in a triangle."""
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_jacobian(
            self._area_coordinates(barycentric_coordinates, triangle), b, c
        )

    def shape_function_hessian_vectorized(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Matrix:
        """Computes jacobian of the shape function in a triangle."""
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_hessian_vectorized(
            self._area_coordinates(barycentric_coordinates, triangle), b, c
        )

    def shape_function_on_edge_left(self, edge: Edge, t: float):
        """Computes shape function on an edge for the left endpoint."""
        triangle = (edge[0], edge[1], self.opposite_vertices[edge][0])
        l1 = self._area_coordinates((t, 1 - t, 0), triangle)[0]
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_on_edge_left(l1, b[2], c[2])

    def shape_function_on_edge_right(self, edge: Edge, t: float):
        """Computes shape function on an edge for the right endpoint."""
        triangle = (edge[0], edge[1], self.opposite_vertices[edge][0])
        l1 = self._area_coordinates((t, 1 - t, 0), triangle)[0]
        _, b, c, _ = self._triangle_parameters(triangle)
        return shape_function_on_edge_right(l1, b[1], c[1])

    def _triangle_contains_point(self, triangle: Triangle, p: Vector) -> bool:
        p1, p2, p3 = (self.points[i] for i in triangle)
        d1 = _sign(p, p1, p2)
        d2 = _sign(p, p2, p3)
        d3 = _sign(p, p3, p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _to_barycentric_coordinates(
        self, triangle: Triangle, p: Vector
    ) -> BarycentricCoordinates:
        """Assuming that the point is inside the triangle, computes its barycentric coordinates."""
        p1, p2, p3 = (self.points[i] for i in triangle)
        v1 = p2 - p1
        v2 = p3 - p1
        v3 = p - p1

        d11 = v1.dot(v1)
        d12 = v1.dot(v2)
        d22 = v2.dot(v2)
        d31 = v3.dot(v1)
        d32 = v3.dot(v2)

        denom = d11 * d22 - d12 * d12
        v = (d22 * d31 - d12 * d32) / denom
        w = (d11 * d32 - d12 * d31) / denom

        if v < 0 or v > 1 or w < 0 or w > 1:
            raise ValueError(
                "Cannot determine barycentric coordinates for a point outside the triangle."
            )
        u = 1 - v - w
        return (u, v, w)

    def _to_barycentric_point(self, p: Vector) -> BarycentricPoint | None:
        """Given cartesian point, determine its containing triangle and barycentric coordinates.

        If the point is outside all grid triangles return None"""
        for triangle in self.triangles:
            if self._triangle_contains_point(triangle, p):
                return (triangle, self._to_barycentric_coordinates(triangle, p))
        return None

    def to_barycentric_curve(self, curve: Curve) -> BarycentricCurve | None:
        """Transform a cartesian curve into a domain curve.

        If any point of the cartesian curve is outside of all grid triangles, return None.
        """
        barycentric_curve: BarycentricCurve = []
        for p in curve:
            p_barycentric = self._to_barycentric_point(p)
            if p_barycentric is None:
                return None
            barycentric_curve.append(p_barycentric)
        return barycentric_curve

    @abstractmethod
    def generate_deformation_curves(
        self,
        num_points: int,
        num_curves_horizontal: int = 0,
        num_curves_vertical: int | None = None,
    ) -> list[Curve]:
        """Generate curves inside the grid domain."""


def _sign(p1: Vector, p2: Vector, p3: Vector):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


class SquareEquilateralGrid(Grid):
    """Grid of equilateral right triangles inside a unit square."""

    def __init__(self, num_horizontal_points: int, fix: str = "none"):
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
            points,
        )

    def generate_deformation_curves(
        self,
        num_points: int,
        num_curves_horizontal: int = 0,
        num_curves_vertical: int | None = None,
    ) -> list[Curve]:
        if num_curves_vertical is None:
            num_curves_vertical = num_curves_horizontal
        cartesian_domain_curves = []
        eps = 1 / (num_points - 1)
        # bottom side
        cartesian_domain_curves.append(
            [Vector([i * eps, 0.0]) for i in range(num_points)]
        )
        # right side
        cartesian_domain_curves.append(
            [Vector([1.0, i * eps]) for i in range(num_points)]
        )
        # top side
        cartesian_domain_curves.append(
            [Vector([i * eps, 1.0]) for i in range(num_points)]
        )
        # left side
        cartesian_domain_curves.append(
            [Vector([0.0, i * eps]) for i in range(num_points)]
        )

        # additional horizontal curves
        for i in range(1, num_curves_horizontal + 1):
            cartesian_domain_curves.append(
                [
                    Vector([j * eps, i / (num_curves_horizontal + 1)])
                    for j in range(num_points)
                ]
            )

        # additional vertical curves
        for i in range(1, num_curves_vertical + 1):
            cartesian_domain_curves.append(
                [
                    Vector([i / (num_curves_horizontal + 1), j * eps])
                    for j in range(num_points)
                ]
            )

        return cartesian_domain_curves
