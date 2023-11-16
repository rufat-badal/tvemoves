""""Implementation of the domain class that in particular can create grids"""

import typing
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tvemoves_rufbad.tensors import Vector

Vertex = int
Edge = tuple[Vertex, Vertex]
Triangle = tuple[Vertex, Vertex, Vertex]


class BarycentricCoordinates:
    """Barycentric coordinates"""

    def __init__(self, u: float, v: float):
        if u < 0 or v < 0 or 1 - u - v < 0:
            raise ValueError("Invalid first two barycentric coordinates provided")
        self.u = u
        self.u = v
        self.w = 1 - u - v


@dataclass
class BarycentricPoint:
    """Barycentric encoding of a point contained in of the grid triangles"""

    triangle: Triangle
    coordinates: BarycentricCoordinates


Curve = list[BarycentricPoint]


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


class Domain(ABC):
    """Interface of a domain"""

    @abstractmethod
    def create_grid(self, scale: float) -> Grid:
        """ "Create a grid adapted to the current domain."""

    @abstractmethod
    def create_curves(
        self,
        segment_length: float,
        num_horizontal_curves: int = 0,
        num_vertical_curves: int | None = None,
    ) -> list[Curve]:
        """Creates a list of discrete curves covered by the grid triangles.

        Boundary curves are always generated. The distance between successive
        points of each curve must be equal to `segment_length`.
        """


FixOption = typing.Literal[None, "all", "lower", "right", "upper", "left"]


class Rectangle(Domain):
    """ "Rectangular reference domain."""

    def __init__(
        self,
        lower_left_corner: Vector = Vector([0.0, 0.0]),
        upper_right_corner: Vector = Vector([1.0, 1.0]),
        fix: FixOption = None,
    ):
        if (
            upper_right_corner[0] < lower_left_corner[0]
            or upper_right_corner[1] < lower_left_corner[1]
        ):
            raise ValueError("Invalid corners provided")
        self.lower_left_corner = lower_left_corner
        self.upper_right_corner = upper_right_corner
        self.width = upper_right_corner[0] - lower_left_corner[0]
        self.height = upper_right_corner[1] - lower_left_corner[1]
        self.fix = fix

    def create_grid(self, scale: float) -> Grid:
        """Return a grid of equilateral right triangles inside the rectangle.

        If `scale` does not evenly divide the width or height of the rectangle (only) the rightmost
        or uppermost edges of the grid might not align exactly with the rectangle boundary.
        """
        num_horizontal_points = self.width // scale
        num_vertical_points = self.height // scale

        def pair_to_vertex(i, j):
            # left to right, bottom to top
            return j * num_horizontal_points + i

        vertices = list(range(num_horizontal_points * num_vertical_points))

        horizontal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j))
            for i in range(num_horizontal_points - 1)
            for j in range(num_vertical_points)
        ]
        vertical_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i, j + 1))
            for i in range(num_horizontal_points)
            for j in range(num_vertical_points - 1)
        ]
        diagonal_edges = [
            (pair_to_vertex(i, j), pair_to_vertex(i + 1, j + 1))
            for i in range(num_horizontal_points - 1)
            for j in range(num_vertical_points - 1)
        ]
        edges = horizontal_edges + vertical_edges + diagonal_edges

        lower_triangles = [
            (
                pair_to_vertex(i + 1, j),
                pair_to_vertex(i, j),
                pair_to_vertex(i + 1, j + 1),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_vertical_points - 1)
        ]
        upper_triangles = [
            (
                pair_to_vertex(i, j + 1),
                pair_to_vertex(i + 1, j + 1),
                pair_to_vertex(i, j),
            )
            for i in range(num_horizontal_points - 1)
            for j in range(num_vertical_points - 1)
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
            pair_to_vertex(num_horizontal_points - 1, j)
            for j in range(num_vertical_points)
        )
        right_boundary_edges = [
            (
                pair_to_vertex(num_horizontal_points - 1, j),
                pair_to_vertex(num_horizontal_points - 1, j + 1),
            )
            for j in range(num_vertical_points - 1)
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
            pair_to_vertex(0, j) for j in range(num_vertical_points)
        )
        left_boundary_edges = [
            (pair_to_vertex(0, j), pair_to_vertex(0, j + 1))
            for j in range(num_horizontal_points - 1)
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
                    v % num_horizontal_points * scale,
                    v // num_horizontal_points * scale,
                ]
            )
            for v in vertices
        ]

        return Grid(
            boundary,
            edges,
            triangles,
            boundary,
            dirichlet_boundary,
            neumann_boundary,
            points,
        )
