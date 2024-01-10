""""Domain class that in particular can create grids"""

from dataclasses import dataclass
from typing import Literal, Protocol, Dict
from abc import ABC
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import patches
from tvemoves_rufbad.tensors import Vector, BarycentricCoordinates
from tvemoves_rufbad.helpers import axis

PLOT_BORDER = 0.05
PLOT_LINEWIDTH = 1.5
PLOT_VERTEX_SIZE = 15


Vertex = int
Edge = tuple[Vertex, Vertex]
Triangle = tuple[Vertex, Vertex, Vertex]
Curve = list[Vector]
TriangleVertices = tuple[Vector, Vector, Vector]
EdgeVertices = tuple[Vector, Vector]


@dataclass(frozen=True)
class BarycentricPoint:
    """Barycentric encoding of a point contained in of the grid triangles"""

    triangle: Triangle
    coordinates: BarycentricCoordinates


BarycentricCurve = list[BarycentricPoint]


@dataclass(frozen=True)
class Boundary:
    """(Part of) the boundary edges and their vertices of a grid"""

    vertices: list[Vertex]
    edges: list[Edge]


def _sign(triangle_vertices: TriangleVertices):
    p1, p2, p3 = triangle_vertices
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _triangle_contains_point(triangle_vertices: TriangleVertices, p: Vector) -> bool:
    """Determine if a point is contained in a triangle."""
    eps = 1e-10

    p1, p2, p3 = triangle_vertices
    d1 = _sign((p, p1, p2))
    d2 = _sign((p, p2, p3))
    d3 = _sign((p, p3, p1))

    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)

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


def _adjust_xlim(ax, xlim=tuple[float, float]) -> None:
    old_xlim = ax.get_xlim()
    ax.set_xlim(min(old_xlim[0], xlim[0]), max(old_xlim[1], xlim[1]))


def _adjust_ylim(ax, ylim=tuple[float, float]) -> None:
    old_ylim = ax.get_ylim()
    ax.set_ylim(min(old_ylim[0], ylim[0]), max(old_ylim[1], ylim[1]))


class Grid(ABC):
    """Simulation grid"""

    def __init__(
        self,
        vertices: list[Vertex],
        edges: list[Edge],
        triangles: list[Triangle],
        boundary: Boundary,
        dirichlet_boundary: Boundary,
        neumann_boundary: Boundary,
        points: list[Vector],
    ):
        self.vertices = vertices
        self.edges = edges
        self.triangles = triangles
        self.boundary = boundary
        self.dirichlet_boundary = dirichlet_boundary
        self.neumann_boundary = neumann_boundary
        self.points = points
        self._mean_edge_length = None
        if self.edges:
            self._mean_edge_length = sum(
                (self.points[i1] - self.points[i2]).norm() for (i1, i2) in self.edges
            ) / len(self.edges)

    def to_barycentric_point(self, p: Vector) -> BarycentricPoint | None:
        """Given cartesian point, determine its containing triangle and
        barycentric coordinates.

        If the point is outside all grid triangles return None.
        """
        # This can be done more efficiently by walking towards the desired point
        # (needs O(1) access to neighboring triangles).
        # Implement this in case the straightforward search is too slow.
        for triangle in self.triangles:
            i1, i2, i3 = triangle
            triangle_vertices = (self.points[i1], self.points[i2], self.points[i3])
            if _triangle_contains_point(triangle_vertices, p):
                return BarycentricPoint(triangle, _to_barycentric_coordinates(triangle_vertices, p))
        return None

    def _plot_vertices(self, ax) -> None:
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        ax.scatter(x, y, color="black", s=PLOT_VERTEX_SIZE)

    def _plot_edges(self, ax) -> None:
        for edge in self.edges:
            i, j = edge
            p, q = self.points[i], self.points[j]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=PLOT_LINEWIDTH)

    def plot(self, ax):
        """Returns matplotlib plot of the grid."""
        if ax is None:
            ax = axis()

        new_xlim = (-PLOT_BORDER, max(p[0] for p in self.points) + PLOT_BORDER)
        _adjust_xlim(ax, new_xlim)

        new_ylim = (-PLOT_BORDER, max(p[1] for p in self.points) + PLOT_BORDER)
        _adjust_ylim(ax, new_ylim)

        self._plot_vertices(ax)
        self._plot_edges(ax)

        return ax

    def triangle_vertices(self, triangle: Triangle) -> TriangleVertices:
        """Transform triangle indices into points."""
        if triangle not in self.triangles:
            raise ValueError("Provided triangle is not in the list of triangles")

        i1, i2, i3 = triangle
        return (self.points[i1], self.points[i2], self.points[i3])

    def edge_vertices(self, edge: Edge) -> EdgeVertices:
        """Transform triangle indices into points."""
        if edge not in self.edges:
            raise ValueError("Provided edge is not in the list of edges")

        i1, i2 = edge
        return (self.points[i1], self.points[i2])

    def point_to_vertex(self, point: Vector) -> Vertex | None:
        """Return vertex id of the point or None if point is not in the grid."""
        return next(
            (
                i
                for i, grid_point in enumerate(self.points)
                if self._points_coincide(point, grid_point)
            ),
            None,
        )

    def _contains_point(self, point: Vector):
        return self.point_to_vertex(point) is not None

    def _contains_edge(self, edge: Edge):
        return edge in self.edges or (edge[1], edge[0]) in self.edges

    def _contains_triangle(self, triangle: Triangle):
        # Only cyclic permutations allowed.
        # E.g. (2, 4, 7) and (4, 7, 2) represent the same triangle but (2, 4, 7) and (2, 7, 4) not.
        i1, i2, i3 = triangle
        return (
            (i1, i2, i3) in self.triangles
            or (i2, i3, i1) in self.triangles
            or (i3, i1, i2) in self.triangles
        )

    def _points_coincide(self, p1: Vector, p2: Vector) -> bool:
        eps = self._mean_edge_length / 1e7 if self._mean_edge_length is not None else 1e-15
        return (p1 - p2).norm() < eps

    def append_edge(self, edge: Edge) -> None:
        """Append an edge if it is not already present."""
        if edge not in self.edges and (edge[1], edge[0]) not in self.edges:
            # Update mean edge length before adding the edge
            num_edges_old = len(self.edges)
            new_edge_length = (self.points[edge[0]] - self.points[edge[1]]).norm()
            if self._mean_edge_length is None:
                self._mean_edge_length = new_edge_length
            else:
                self._mean_edge_length = self._mean_edge_length * num_edges_old / (
                    num_edges_old + 1
                ) + new_edge_length / (num_edges_old + 1)

            self.edges.append(edge)

    def __eq__(self, other):
        """Equality for two grids. We ignore reordering of vertices, edges, and triangles."""
        if not isinstance(other, Grid):
            return False

        if sorted(self.vertices) != sorted(other.vertices):
            return False

        if len(self.points) != len(other.points):
            return False
        other_to_self_vertex: Dict[Vertex, Vertex] = {}
        for other_vertex, p in enumerate(other.points):
            self_vertex = self.point_to_vertex(p)
            if self_vertex is None:
                return False
            other_to_self_vertex[other_vertex] = self_vertex
        # As things went well up to this point we can assume that other_to_self_vertex is set up.

        if len(self.edges) != len(other.edges):
            return False
        if any(
            not self._contains_edge((other_to_self_vertex[i1], other_to_self_vertex[i2]))
            for (i1, i2) in other.edges
        ):
            return False

        if len(self.triangles) != len(other.triangles):
            return False
        if any(
            not self._contains_triangle(
                (other_to_self_vertex[i1], other_to_self_vertex[i2], other_to_self_vertex[i3])
            )
            for (i1, i2, i3) in other.triangles
        ):
            return False

        if (
            not _boundaries_coincide(self.boundary, other.boundary, other_to_self_vertex)
            or not _boundaries_coincide(
                self.dirichlet_boundary, other.dirichlet_boundary, other_to_self_vertex
            )
            or not _boundaries_coincide(
                self.neumann_boundary, other.neumann_boundary, other_to_self_vertex
            )
        ):
            return False

        return True


def _boundaries_coincide(
    boundary: Boundary, other_boundary: Boundary, other_to_self_vertex: Dict[Vertex, Vertex]
) -> bool:
    if sorted(boundary.vertices) != sorted(
        [other_to_self_vertex[other_vertex] for other_vertex in other_boundary.vertices]
    ):
        return False

    if len(boundary.edges) != len(other_boundary.edges):
        return False
    other_edges = [
        (other_to_self_vertex[i1], other_to_self_vertex[i2]) for (i1, i2) in other_boundary.edges
    ]
    if any(
        edge not in boundary.edges and (edge[1], edge[0]) not in boundary.edges
        for edge in other_edges
    ):
        return False

    return True


_TriangleBarycentricCoords = Dict[
    Triangle, tuple[BarycentricCoordinates, BarycentricCoordinates, BarycentricCoordinates]
]
_EdgeCoords = Dict[Edge, tuple[float, float]]


class RefinedGrid(Grid):
    """Refinement of a grid."""

    def __init__(
        self,
        vertices: list[Vertex],
        edges: list[Edge],
        triangles: list[Triangle],
        boundary: Boundary,
        dirichlet_boundary: Boundary,
        neumann_boundary: Boundary,
        points: list[Vector],
        coarse_grid: Grid,
        coarse_triangle: Dict[Triangle, Triangle],
        coarse_barycentric_coordinates: _TriangleBarycentricCoords,
        coarse_edge: Dict[Edge, Edge],
        coarse_edge_coordinates: _EdgeCoords,
    ):
        super().__init__(
            vertices,
            edges,
            triangles,
            boundary,
            dirichlet_boundary,
            neumann_boundary,
            points,
        )
        self._coarse_grid = coarse_grid
        self._coarse_triangle = coarse_triangle
        self._coarse_barycentric_coordinates = coarse_barycentric_coordinates
        self._coarse_edge = coarse_edge
        self._coarse_edge_coordinates = coarse_edge_coordinates

    def coarse(self) -> Grid:
        """Return the coarse grid for which self is a refinedment."""
        return self._coarse_grid

    def to_coarse_barycentric_point(self, fine_point: BarycentricPoint) -> BarycentricPoint:
        """Transform a barycentric point in the fine mesh to a barycentric point in the coarse mesh"""
        fine_triangle = fine_point.triangle
        i1, i2, i3 = fine_triangle
        if (i1, i2, i3) not in self.triangles:
            if (i2, i3, i1) in self.triangles:
                fine_triangle = (i2, i3, i1)
            elif (i3, i1, i2) in self.triangles:
                fine_triangle = (i3, i1, i2)
            else:
                raise ValueError("triangle does not appear in the grid")
        coarse_triangle = self._coarse_triangle[fine_triangle]
        p = fine_point.coordinates
        q1, q2, q3 = self._coarse_barycentric_coordinates[fine_triangle]
        coarse_barycentric_coordinates = BarycentricCoordinates(
            p.l1 * q1.l1 + p.l2 * q2.l1 + p.l3 * q3.l1, p.l1 * q1.l2 + p.l2 * q2.l2 + p.l3 * q3.l2
        )
        return BarycentricPoint(coarse_triangle, coarse_barycentric_coordinates)

    def to_coarse_edge_point(self, fine_edge: Edge, t: float) -> tuple[Edge, float]:
        """Transform relative point on a fine edge to a relative point on the repsective coarse edge.

        This does only work for fine edges that lie on a coarse edge. Else an error is raised.
        """

        if fine_edge not in self._coarse_edge or fine_edge not in self._coarse_edge_coordinates:
            raise ValueError("only fine edges that lie on coarse edges are valid inputs")

        coarse_edge = self._coarse_edge[fine_edge]
        t1, t2 = self._coarse_edge_coordinates[fine_edge]
        coarse_t = t * t1 + (1 - t) * t2
        return coarse_edge, coarse_t


class Domain(Protocol):
    """Abstract domain"""

    def grid(self, scale: float) -> Grid:
        """Create a grid adapted to the current domain."""

    def curves(
        self,
        num_points_per_curve: int,
        num_horizontal_curves: int = 0,
        num_vertical_curves: int | None = None,
    ) -> list[Curve]:
        """Creates a list of discrete curves inside the domain.

        Boundary curves are always generated. Each curve contains num_points_per_curve
        points.
        """

    def plot(self, ax):
        """Visualize the domain."""

    def refine(self, grid: Grid, refinement_factor: int) -> RefinedGrid:
        """Refine a provided grid. It is the responsibilty of the caller to only provide grids that
        were generated by self.grid()."""


FixOption = Literal["lower", "right", "upper", "left"]


class RectangleDomain(Domain):
    """Rectangular reference domain."""

    def __init__(
        self,
        width: float,
        height: float,
        fix: None | FixOption | list[FixOption] = None,
    ):
        self.width = width
        self.height = height
        self.fix = fix

    def grid(self, scale: float) -> Grid:
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
                pair_to_vertex(i + 1, j + 1),
                pair_to_vertex(i, j + 1),
                pair_to_vertex(i, j),
            )
            for i in range(num_horizontal_vertices - 1)
            for j in range(num_vertical_vertices - 1)
        ]
        triangles = lower_triangles + upper_triangles

        lower_boundary_vertices = set(pair_to_vertex(i, 0) for i in range(num_horizontal_vertices))
        lower_boundary_edges = [
            (pair_to_vertex(i, 0), pair_to_vertex(i + 1, 0))
            for i in range(num_horizontal_vertices - 1)
        ]
        right_boundary_vertices = set(
            pair_to_vertex(num_horizontal_vertices - 1, j) for j in range(num_vertical_vertices)
        )
        right_boundary_edges = [
            (
                pair_to_vertex(num_horizontal_vertices - 1, j),
                pair_to_vertex(num_horizontal_vertices - 1, j + 1),
            )
            for j in range(num_vertical_vertices - 1)
        ]
        upper_boundary_vertices = set(
            pair_to_vertex(i, num_vertical_vertices - 1) for i in range(num_horizontal_vertices)
        )
        upper_boundary_edges = [
            (
                pair_to_vertex(i, num_vertical_vertices - 1),
                pair_to_vertex(i + 1, num_vertical_vertices - 1),
            )
            for i in range(num_horizontal_vertices - 1)
        ]
        left_boundary_vertices = set(pair_to_vertex(0, j) for j in range(num_vertical_vertices))
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
            lower_boundary_edges + right_boundary_edges + upper_boundary_edges + left_boundary_edges
        )
        boundary = Boundary(boundary_vertices, boundary_edges)

        dirichlet_vertices = set()
        dirichlet_edges = set()
        neumann_vertices = set(boundary_vertices)
        neumann_edges = set(boundary_edges)

        if self.fix is not None:
            if not isinstance(self.fix, list):
                fixes = [self.fix]
            else:
                fixes = list(set(self.fix))
            for fix in fixes:
                match fix:
                    case "lower":
                        dirichlet_vertices.update(lower_boundary_vertices)
                        dirichlet_edges.update(lower_boundary_edges)
                        neumann_vertices.difference_update(lower_boundary_vertices)
                        neumann_edges.difference_update(lower_boundary_edges)
                    case "right":
                        dirichlet_vertices.update(right_boundary_vertices)
                        dirichlet_edges.update(right_boundary_edges)
                        neumann_vertices.difference_update(right_boundary_vertices)
                        neumann_edges.difference_update(right_boundary_edges)
                    case "upper":
                        dirichlet_vertices.update(upper_boundary_vertices)
                        dirichlet_edges.update(upper_boundary_edges)
                        neumann_vertices.difference_update(upper_boundary_vertices)
                        neumann_edges.difference_update(upper_boundary_edges)
                    case "left":
                        dirichlet_vertices.update(left_boundary_vertices)
                        dirichlet_edges.update(left_boundary_edges)
                        neumann_vertices.difference_update(left_boundary_vertices)
                        neumann_edges.difference_update(left_boundary_edges)

        dirichlet_boundary = Boundary(list(dirichlet_vertices), list(dirichlet_edges))
        neumann_boundary = Boundary(list(neumann_vertices), list(neumann_edges))

        points = [
            Vector([
                v % num_horizontal_vertices * scale,
                v // num_horizontal_vertices * scale,
            ])
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

    def curves(
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
                Vector([
                    i * self.width / (num_points_per_curve - 1),
                    j * self.height / (num_all_horizontal_curves - 1),
                ])
                for i in range(num_points_per_curve)
            ]
            for j in range(num_all_horizontal_curves)
        ]

        vertical_curves = [
            [
                Vector([
                    j * self.width / (num_all_vertical_curves - 1),
                    i * self.height / (num_points_per_curve - 1),
                ])
                for i in range(num_points_per_curve)
            ]
            for j in range(num_all_vertical_curves)
        ]

        return horizontal_curves + vertical_curves

    def plot(self, ax):
        if ax is None:
            ax = axis()

        plt.xlim(-PLOT_BORDER, self.width + PLOT_BORDER)
        plt.ylim(-PLOT_BORDER, self.height + PLOT_BORDER)

        rectangle = patches.Rectangle(
            (0, 0),
            self.width,
            self.height,
            edgecolor="blue",
            facecolor="lightblue",
            linewidth=PLOT_LINEWIDTH,
        )
        ax.add_patch(rectangle)

        return ax

    def refine(self, grid: Grid, refinement_factor: int) -> RefinedGrid:
        if refinement_factor < 1:
            raise ValueError(
                f"only positive refinement factors allowed, but {refinement_factor} was provided"
            )

        intermediate_grid = Grid(
            deepcopy(grid.vertices),
            [],
            [],
            Boundary(deepcopy(grid.boundary.vertices), []),
            Boundary(deepcopy(grid.dirichlet_boundary.vertices), []),
            Boundary(deepcopy(grid.neumann_boundary.vertices), []),
            deepcopy(grid.points),
        )

        coarse_triangle: Dict[Triangle, Triangle] = {}
        coarse_barycentric_coordinates: _TriangleBarycentricCoords = {}
        coarse_edge: Dict[Edge, Edge] = {}
        coarse_edge_coordinates: _EdgeCoords = {}

        for triangle in grid.triangles:
            _refine_equilateral_triangle(
                triangle,
                grid,
                refinement_factor,
                intermediate_grid,
                coarse_triangle,
                coarse_barycentric_coordinates,
                coarse_edge,
                coarse_edge_coordinates,
            )

        return RefinedGrid(
            intermediate_grid.vertices,
            intermediate_grid.edges,
            intermediate_grid.triangles,
            intermediate_grid.boundary,
            intermediate_grid.dirichlet_boundary,
            intermediate_grid.neumann_boundary,
            intermediate_grid.points,
            grid,
            coarse_triangle,
            coarse_barycentric_coordinates,
            coarse_edge,
            coarse_edge_coordinates,
        )


def _refine_equilateral_triangle(
    triangle: Triangle,
    grid: Grid,
    refinement_factor: int,
    intermediate_grid: Grid,
    coarse_triangle: Dict[Triangle, Triangle],
    coarse_barycentric_coordinates: _TriangleBarycentricCoords,
    coarse_edge: Dict[Edge, Edge],
    coarse_edge_coordinates: _EdgeCoords,
) -> None:
    """Refine a single equlateral triangle of the coarse grid. This function modifies refined_grid!

    It is the responsibility of the caller to only provide triangles in grid."""

    i1, i2, i3 = triangle
    p1, p2, p3 = grid.triangle_vertices(triangle)
    eps = 1e-15
    assert abs(p1[1] - p2[1]) < eps, "First edge of an equilateral triangle must be horizontal"
    assert abs(p2[0] - p3[0]) < eps, "Second edge of an equilateral triangle must be vertical"

    # Create new vertex indices and points
    next_vertex = intermediate_grid.vertices[-1] + 1
    triangle_vertices: Dict[tuple[int, int], Vertex] = {}
    triangle_vertices[(0, 0)] = i1
    triangle_vertices[(refinement_factor, 0)] = i2
    triangle_vertices[(0, refinement_factor)] = i3
    triangle_points: Dict[tuple[int, int], Vector] = {}
    triangle_points[(0, 0)] = p1
    triangle_points[(refinement_factor, 0)] = p2
    triangle_points[(0, refinement_factor)] = p3
    for l in range(refinement_factor + 1):
        for k in range(refinement_factor + 1 - l):
            # ignore endpoints
            if k == 0 and l == 0:
                continue
            if k == refinement_factor and l == 0:
                continue
            if k == 0 and l == refinement_factor:
                continue
            next_barycentric_coords = BarycentricCoordinates(
                1 - k / refinement_factor - l / refinement_factor, k / refinement_factor
            )
            next_barycentric_coords.assert_bounds()
            next_point = (
                next_barycentric_coords.l1 * p1
                + next_barycentric_coords.l2 * p2
                + next_barycentric_coords.l3 * p3
            )
            triangle_points[(k, l)] = next_point
            # Check if the vertex was already added in a previous refinement step
            old_vertex = intermediate_grid.point_to_vertex(next_point)
            if old_vertex is None:
                triangle_vertices[(k, l)] = next_vertex
                intermediate_grid.vertices.append(next_vertex)
                intermediate_grid.points.append(next_point)
                next_vertex += 1
            else:
                triangle_vertices[(k, l)] = old_vertex

    # Add lower triangles
    # Refined triangles are unique for each refinement step!
    for l in range(refinement_factor):
        for k in range(refinement_factor - l):
            next_triangle = (
                triangle_vertices[(k, l)],
                triangle_vertices[(k + 1, l)],
                triangle_vertices[(k, l + 1)],
            )
            p1 = intermediate_grid.points[triangle_vertices[(k, l)]]
            p2 = intermediate_grid.points[triangle_vertices[(k + 1, l)]]
            intermediate_grid.triangles.append(next_triangle)
            coarse_triangle[next_triangle] = triangle
            coarse_barycentric_coordinates[next_triangle] = (
                BarycentricCoordinates(
                    1 - k / refinement_factor - l / refinement_factor, k / refinement_factor
                ),
                BarycentricCoordinates(
                    1 - (k + 1) / refinement_factor - l / refinement_factor,
                    (k + 1) / refinement_factor,
                ),
                BarycentricCoordinates(
                    1 - k / refinement_factor - (l + 1) / refinement_factor, k / refinement_factor
                ),
            )

    # Add upper triangles
    for l in range(refinement_factor - 1):
        for k in range(1, refinement_factor - l):
            next_triangle = (
                triangle_vertices[(k, l + 1)],
                triangle_vertices[(k - 1, l + 1)],
                triangle_vertices[(k, l)],
            )
            intermediate_grid.triangles.append(next_triangle)
            coarse_triangle[next_triangle] = triangle
            coarse_barycentric_coordinates[next_triangle] = (
                BarycentricCoordinates(
                    1 - k / refinement_factor - (l + 1) / refinement_factor, k / refinement_factor
                ),
                BarycentricCoordinates(
                    1 - (k - 1) / refinement_factor - (l + 1) / refinement_factor,
                    (k - 1) / refinement_factor,
                ),
                BarycentricCoordinates(
                    1 - k / refinement_factor - l / refinement_factor, k / refinement_factor
                ),
            )

    # The edge orientation induced by the triangle orientation might not coincide with the real edge
    # orientation of the coarse grid.
    horizontal_edge_correct = (i1, i2) in grid.edges
    vertical_edge_correct = (i2, i3) in grid.edges
    diagonal_edge_correct = (i3, i1) in grid.edges

    def assure_orientation(coarse_orientation_correct: bool, ordering):
        if isinstance(ordering, tuple):
            return ordering if coarse_orientation_correct else tuple(reversed(ordering))
        if isinstance(ordering, list):
            return ordering if coarse_orientation_correct else list(reversed(ordering))

    for l in range(refinement_factor):
        for k in range(refinement_factor - l):
            # Fine edges have the same orientation as the coarse edge in the triangle parallel to them.
            # Add horizontal edges
            intermediate_grid.append_edge(
                assure_orientation(
                    horizontal_edge_correct,
                    (triangle_vertices[(k, l)], triangle_vertices[(k + 1, l)]),
                )
            )
            # Add vertical edges
            intermediate_grid.append_edge(
                assure_orientation(
                    vertical_edge_correct,
                    (triangle_vertices[(k + 1, l)], triangle_vertices[(k, l + 1)]),
                )
            )
            # Add diagonal edges
            intermediate_grid.append_edge(
                assure_orientation(
                    diagonal_edge_correct,
                    (triangle_vertices[(k, l)], triangle_vertices[(k, l + 1)]),
                )
            )

    horizontal_edge_coarse = assure_orientation(horizontal_edge_correct, (i1, i2))
    vertical_edge_coarse = assure_orientation(vertical_edge_correct, (i2, i3))
    diagonal_edge_coarse = assure_orientation(diagonal_edge_correct, (i3, i1))

    horizontal_edge_vertices = assure_orientation(
        horizontal_edge_correct, [triangle_vertices[(i, 0)] for i in range(refinement_factor + 1)]
    )
    vertical_edge_vertices = assure_orientation(
        vertical_edge_correct,
        [triangle_vertices[(refinement_factor - i, i)] for i in range(refinement_factor + 1)],
    )

    for edge, edge_vertices in zip(
        [horizontal_edge_coarse, vertical_edge_coarse],
        [horizontal_edge_vertices, vertical_edge_vertices],
    ):
        # Boundary vertices and edges are unique for each refinement step!
        _refine_boundary_edge(edge, edge_vertices, grid.boundary, intermediate_grid.boundary)
        _refine_boundary_edge(
            edge, edge_vertices, grid.dirichlet_boundary, intermediate_grid.dirichlet_boundary
        )
        _refine_boundary_edge(
            edge, edge_vertices, grid.neumann_boundary, intermediate_grid.neumann_boundary
        )

    def assure_orientation_and_weights(
        coarse_orientation_correct: bool, pair: tuple[float, float]
    ) -> tuple[float, float]:
        return pair if coarse_orientation_correct else (1 - pair[1], 1 - pair[0])

    for i in range(refinement_factor):
        horizontal_edge_fine = assure_orientation(
            horizontal_edge_correct, (triangle_vertices[(i, 0)], triangle_vertices[(i + 1, 0)])
        )
        coarse_edge[horizontal_edge_fine] = horizontal_edge_coarse
        coarse_edge_coordinates[horizontal_edge_fine] = assure_orientation_and_weights(
            horizontal_edge_correct,
            (
                1 - i / refinement_factor,
                1 - (i + 1) / refinement_factor,
            ),
        )

        vertical_edge_fine = assure_orientation(
            vertical_edge_correct,
            (
                triangle_vertices[(refinement_factor - i, i)],
                triangle_vertices[(refinement_factor - (i + 1), i + 1)],
            ),
        )
        coarse_edge[vertical_edge_fine] = vertical_edge_coarse
        coarse_edge_coordinates[vertical_edge_fine] = assure_orientation_and_weights(
            vertical_edge_correct,
            (
                1 - i / refinement_factor,
                1 - (i + 1) / refinement_factor,
            ),
        )

        diagonal_edge_fine = assure_orientation(
            diagonal_edge_correct,
            (
                triangle_vertices[(0, refinement_factor - i)],
                triangle_vertices[(0, refinement_factor - (i + 1))],
            ),
        )
        coarse_edge[diagonal_edge_fine] = diagonal_edge_coarse
        coarse_edge_coordinates[diagonal_edge_fine] = assure_orientation_and_weights(
            diagonal_edge_correct,
            (
                1 - i / refinement_factor,
                1 - (i + 1) / refinement_factor,
            ),
        )


def _refine_boundary_edge(
    edge: Edge,
    edge_vertices: list[Vertex],
    boundary: Boundary,
    intermediate_boundary: Boundary,
) -> None:
    if edge not in boundary.edges:
        return

    intermediate_boundary.vertices.extend(edge_vertices[1:-1])
    intermediate_boundary.edges.extend(
        (i, j) for i, j in zip(edge_vertices[:-1], edge_vertices[1:])
    )
