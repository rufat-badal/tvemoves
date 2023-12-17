"""Module providing interpolations in triangles."""

from typing import Protocol
from tvemoves_rufbad.tensors import Vector, BarycentricCoordinates, Matrix, Tensor3D
from tvemoves_rufbad.domain import BarycentricPoint, Grid, Triangle, Edge, RefinedGrid
from tvemoves_rufbad.bell_finite_element import (
    bell_interpolation,
    bell_interpolation_gradient,
    bell_interpolation_hessian,
    bell_interpolation_on_edge,
    transform_gradient,
)


class Interpolation(Protocol):
    """Interpolation protocol."""

    def __call__(self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates):
        """Compute the value of the interpolation in a triangle."""

    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Vector:
        """Compute the gradient of the interpolation in a triangle."""

    def on_edge(self, edge: Edge, t: float):
        """Computes the value of the interpolation on an edge."""

    def hessian(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Matrix:
        """Compute the hessian of the interpolation in a triangle."""


class P1Interpolation(Interpolation):
    """Piecewise affine interpolation."""

    def __init__(self, grid: Grid, params: list):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ):
        i1, i2, i3 = triangle
        l1, l2, l3 = barycentric_coordinates
        return l1 * self._params[i1] + l2 * self._params[i2] + l3 * self._params[i3]

    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Vector:
        # Unused argument 'barycentric_coordinates' assures the same calling convention
        # as in the case of the C1 interpolation
        i1, i2, i3 = triangle
        barycentric_gradient = Vector([self._params[i1], self._params[i2], self._params[i3]])
        return transform_gradient(self._grid.triangle_vertices(triangle), barycentric_gradient)

    def on_edge(self, edge: Edge, t: float):
        i1, i2 = edge
        return t * self._params[i1] + (1 - t) * self._params[i2]

    def hessian(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Matrix:
        raise NotImplementedError("Hessian cannot be computed for P1 interpolations")


class C1Interpolation(Interpolation):
    """Interpolation using degree 5 polynomials assuring C1 regularity along the edges."""

    def __init__(self, grid: Grid, params: list[list]):
        """params is a list of C1 parameters. Its length is equal to the number of
        grid vertices.

        Each C1 parameter is a list of 6 values corresponding to
        u, u_x, u_y, u_xx, u_xy, u_yy,
        where u denotes the function wish to interpolate.
        """
        if any(len(p) != 6 for p in params):
            raise ValueError("each parameter provided must be a list of 6 elements")
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")

        self._grid = grid
        self._params = params

    def _triangle_params(self, triangle: Triangle) -> Vector:
        i1, i2, i3 = triangle
        params_vec1 = Vector(self._params[i1])
        params_vec2 = Vector(self._params[i2])
        params_vec3 = Vector(self._params[i3])
        return params_vec1.extend(params_vec2).extend(params_vec3)

    def __call__(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ):
        """Compute the value of the interpolation in a triangle."""
        return bell_interpolation(
            self._grid.triangle_vertices(triangle),
            barycentric_coordinates,
            self._triangle_params(triangle),
        )

    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Vector:
        """Computes gradient of the interpolation."""
        return bell_interpolation_gradient(
            self._grid.triangle_vertices(triangle),
            barycentric_coordinates,
            self._triangle_params(triangle),
        )

    def hessian(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Matrix:
        return bell_interpolation_hessian(
            self._grid.triangle_vertices(triangle),
            barycentric_coordinates,
            self._triangle_params(triangle),
        )

    def on_edge(self, edge: Edge, t: float):
        i1, i2 = edge
        params_vec1 = Vector(self._params[i1])
        params_vec2 = Vector(self._params[i2])
        edge_params = params_vec1.extend(params_vec2)
        edge_vertices = (self._grid.points[i1], self._grid.points[i2])
        return bell_interpolation_on_edge(edge_vertices, t, edge_params)


class RefinedInterpolation(Interpolation):
    """Refinement of an interpolation.

    The interpolation stays the same but is extrapolated to a finer grid.
    """

    def __init__(self, coarse_interpolation: Interpolation, refined_grid: RefinedGrid):
        """Extends coarse_interpolation to a finer grid.

        It is the responsibility of the caller that refined_grid.coarse() coincides exactly with
        coarse_interpolation._grid.
        No permutation of indices allowed!
        """

        self._coarse_interpolation = coarse_interpolation
        self._refined_grid = refined_grid

    def _to_coarse_parameters(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> tuple[Triangle, BarycentricCoordinates]:
        bp_fine = BarycentricPoint(triangle, barycentric_coordinates)
        bp_coarse = self._refined_grid.to_coarse_barycentric_point(bp_fine)
        return bp_coarse.triangle, bp_coarse.coordinates

    def __call__(self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates):
        return self._coarse_interpolation(
            *self._to_coarse_parameters(triangle, barycentric_coordinates)
        )

    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Vector:
        return self._coarse_interpolation.gradient(
            *self._to_coarse_parameters(triangle, barycentric_coordinates)
        )

    def on_edge(self, edge: Edge, t: float):
        """This function is only defined on edges that lie on a corase edge.

        Otherwise an error is raised.
        """

    def hessian(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Matrix:
        return self._coarse_interpolation.hessian(
            *self._to_coarse_parameters(triangle, barycentric_coordinates)
        )


class Deformation:
    """Interpolation of a deformation."""

    def __init__(self, y1_interpolation: Interpolation, y2_interpolation: Interpolation):
        self._y = [y1_interpolation, y2_interpolation]

    def __getitem__(self, i: int):
        """Access the component interpolations directly."""
        return self._y[i]

    def __call__(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Vector:
        return Vector(
            [
                self._y[0](triangle, barycentric_coordinates),
                self._y[1](triangle, barycentric_coordinates),
            ]
        )

    def strain(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ) -> Matrix:
        """Compute the strain of the deformation in a triangle."""
        return (
            self._y[0]
            .gradient(triangle, barycentric_coordinates)
            .stack(self._y[1].gradient(triangle, barycentric_coordinates))
        )

    def on_edge(self, edge: Edge, t: float):
        """Compute the deformation on an edge."""
        return Vector(
            [
                self._y[0].on_edge(edge, t),
                self._y[1].on_edge(edge, t),
            ]
        )

    def hyper_strain(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Tensor3D:
        """Compute the hyper strain of the deformation in a triangle."""
        hessian_y1 = self._y[0].hessian(triangle, barycentric_coordinates)
        hessian_y2 = self._y[1].hessian(triangle, barycentric_coordinates)
        return hessian_y1.stack(hessian_y2)
