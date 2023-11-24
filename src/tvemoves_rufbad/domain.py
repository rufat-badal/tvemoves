""""Domain class that in particular can create grids"""

import typing
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tvemoves_rufbad.tensors import Vector


class BarycentricCoordinates:
    """Barycentric coordinates"""

    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v
        self.w = 1 - u - v

    def __str__(self) -> str:
        return f"({self.u}, {self.v}, {self.w})"

    def __repr__(self) -> str:
        return f"BarycentricCoordinates{self.__str__()}"


Vertex = int
Edge = tuple[Vertex, Vertex]
Triangle = tuple[Vertex, Vertex, Vertex]


@dataclass
class BarycentricPoint:
    """Barycentric encoding of a point contained in of the grid triangles"""

    triangle: Triangle
    coordinates: BarycentricCoordinates


Curve = list[Vector]
BarycentricCurve = list[BarycentricPoint]


@dataclass(frozen=True)
class Boundary:
    """(Part of) the boundary edges and their vertices of a grid"""

    vertices: list[Vertex]
    edges: list[Edge]


@dataclass(frozen=True)
class Grid:
    """Simulation grid"""

    vertices: list[Vertex]
    edges: list[Edge]
    triangles: list[Triangle]
    boundary: Boundary
    dirichlet_boundary: Boundary
    neumann_boundary: Boundary
    points: list[Vector]


TriangleVertices = tuple[Vector, Vector, Vector]


def _sign(triangle_vertices: TriangleVertices):
    p1, p2, p3 = triangle_vertices
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _triangle_contains_point(triangle_vertices: TriangleVertices, p: Vector) -> bool:
    """Determine if a point is contained in a triangle."""
    p1, p2, p3 = triangle_vertices
    d1 = _sign((p, p1, p2))
    d2 = _sign((p, p2, p3))
    d3 = _sign((p, p3, p1))

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def _to_barycentric_coordinates(
    triangle_vertices: TriangleVertices, p: Vector
) -> BarycentricCoordinates:
    """Assuming that the point is inside the triangle, computes its barycentric coordinates."""
    p1, p2, p3 = triangle_vertices

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
    u = 1 - v - w

    return BarycentricCoordinates(u, v)


def _to_barycentric_point(p: Vector, grid: Grid) -> BarycentricPoint | None:
    """Given cartesian point and a grid, determine its containing triangle and
    barycentric coordinates.

    If the point is outside all grid triangles return None.
    """
    for triangle in grid.triangles:
        i1, i2, i3 = triangle
        triangle_vertices = (grid.points[i1], grid.points[i2], grid.points[i3])
        if _triangle_contains_point(triangle_vertices, p):
            return BarycentricPoint(
                triangle, _to_barycentric_coordinates(triangle_vertices, p)
            )
    return None


def to_barycentric_curve(curve: Curve, grid: Grid) -> BarycentricCurve:
    """Transform a curve of Euclidean points to a barycentric curve.

    Points that are not contained in the respective grid are removed from the returned curve.
    """
    barycentric_curve: BarycentricCurve = []
    for p in curve:
        p_barycentric = _to_barycentric_point(p, grid)
        if p_barycentric is not None:
            barycentric_curve.append(p_barycentric)
    return barycentric_curve


class Domain(ABC):
    """Interface of a domain"""

    @abstractmethod
    def create_grid(self, scale: float) -> Grid:
        """ "Create a grid adapted to the current domain."""

    @abstractmethod
    def create_curves(
        self,
        num_points_per_curve: int,
        num_horizontal_curves: int = 0,
        num_vertical_curves: int | None = None,
    ) -> list[Curve]:
        """Creates a list of discrete curves inside the domain.

        Boundary curves are always generated. Each curve contains num_points_per_curve
        points.
        """


FixOption = typing.Literal[None, "all", "lower", "right", "upper", "left"]


class RectangleDomain(Domain):
    """Rectangular reference domain."""

    def __init__(
        self,
        width: float,
        height: float,
        fix: FixOption = None,
    ):
        self.width = width
        self.height = height
        self.fix = fix

    def create_grid(self, scale: float) -> Grid:
        """Return a grid of equilateral right triangles inside the rectangle.

        If `scale` does not evenly divide the width or height of the rectangle (only) the rightmost
        or uppermost edges of the grid might not align exactly with the rectangle boundary.
        """
        num_horizontal_vertices = int(self.width / scale) + 1
        num_vertical_vertices = int(self.height / scale) + 1

        def pair_to_vertex(i, j):
            # left to right, bottom to top
            return j * num_horizontal_vertices + i

        vertices = list(range(num_horizontal_vertices * num_vertical_vertices))

        horizontal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j))
            for j in range(num_vertical_vertices)
            for i in range(num_horizontal_vertices - 1)
        ]
        vertical_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i, j + 1))
            for i in range(num_horizontal_vertices)
            for j in range(num_vertical_vertices - 1)
        ]
        diagonal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j + 1))
            for i in range(num_horizontal_vertices - 1)
            for j in range(num_vertical_vertices - 1)
        ]
        edges = horizontal_edges + vertical_edges + diagonal_edges

        lower_triangles = [
            (
                pair_to_vertex(i, j),
                pair_to_vertex(i + 1, j),
                pair_to_vertex(i + 1, j + 1),
            )
            for i in range(num_horizontal_vertices - 1)
            for j in range(num_vertical_vertices - 1)
        ]
        upper_triangles = [
            (
                pair_to_vertex(i, j),
                pair_to_vertex(i + 1, j + 1),
                pair_to_vertex(i, j + 1),
            )
            for i in range(num_horizontal_vertices - 1)
            for j in range(num_vertical_vertices - 1)
        ]
        triangles = lower_triangles + upper_triangles

        lower_boundary_vertices = set(
            pair_to_vertex(i, 0) for i in range(num_horizontal_vertices)
        )
        lower_boundary_edges = [
            (pair_to_vertex(i, 0), pair_to_vertex(i + 1, 0))
            for i in range(num_horizontal_vertices - 1)
        ]
        right_boundary_vertices = set(
            pair_to_vertex(num_horizontal_vertices - 1, j)
            for j in range(num_vertical_vertices)
        )
        right_boundary_edges = [
            (
                pair_to_vertex(num_horizontal_vertices - 1, j),
                pair_to_vertex(num_horizontal_vertices - 1, j + 1),
            )
            for j in range(num_vertical_vertices - 1)
        ]
        upper_boundary_vertices = set(
            pair_to_vertex(i, num_vertical_vertices - 1)
            for i in range(num_horizontal_vertices)
        )
        upper_boundary_edges = [
            (
                pair_to_vertex(i, num_vertical_vertices - 1),
                pair_to_vertex(i + 1, num_vertical_vertices - 1),
            )
            for i in range(num_horizontal_vertices - 1)
        ]
        left_boundary_vertices = set(
            pair_to_vertex(0, j) for j in range(num_vertical_vertices)
        )
        left_boundary_edges = [
            (pair_to_vertex(0, j), pair_to_vertex(0, j + 1))
            for j in range(num_vertical_vertices - 1)
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
        boundary = Boundary(boundary_vertices, boundary_edges)

        match self.fix:
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

        dirichlet_boundary = Boundary(dirichlet_vertices, dirichlet_edges)
        neumann_boundary = Boundary(neumann_vertices, neumann_edges)

        points = [
            Vector(
                [
                    v % num_horizontal_vertices * scale,
                    v // num_horizontal_vertices * scale,
                ]
            )
            for v in vertices
        ]

        return Grid(
            vertices,
            edges,
            triangles,
            boundary,
            dirichlet_boundary,
            neumann_boundary,
            points,
        )

    def create_curves(
        self,
        num_points_per_curve: int,
        num_horizontal_curves: int = 0,
        num_vertical_curves: int | None = None,
    ) -> list[Curve]:
        """Creates horizontal and vertical curves inside the rectangle.

        `num_horizontal_curves` is the number of internal horizontal curves as we always
        generate the boundary curves. Hence, there are `num_horizontal_cruves + 2`
        horizontal curves (similarly for the vertical curves).
        """

        if num_vertical_curves is None:
            num_vertical_curves = num_horizontal_curves

        num_all_horizontal_curves = num_horizontal_curves + 2
        num_all_vertical_curves = num_vertical_curves + 2

        horizontal_curves = [
            [
                Vector(
                    [
                        i * self.width / (num_points_per_curve - 1),
                        j * self.height / (num_all_horizontal_curves - 1),
                    ]
                )
                for i in range(num_points_per_curve)
            ]
            for j in range(num_all_horizontal_curves)
        ]

        vertical_curves = [
            [
                Vector(
                    [
                        j * self.width / (num_all_vertical_curves - 1),
                        i * self.height / (num_points_per_curve - 1),
                    ]
                )
                for i in range(num_points_per_curve)
            ]
            for j in range(num_all_vertical_curves)
        ]

        return horizontal_curves + vertical_curves