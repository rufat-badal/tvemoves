""""Implementation of the domain class that in particular can create grids"""

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


@dataclass
class Grid(frozen=True):
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
