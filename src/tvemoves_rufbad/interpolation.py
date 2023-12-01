"""Module providing interpolations in triangles."""

from abc import ABC, abstractmethod
from tvemoves_rufbad.tensors import Vector, Matrix, Tensor3D
from tvemoves_rufbad.domain import Grid, Triangle, Edge, BarycentricCoordinates
from tvemoves_rufbad.bell_finite_element import (
    transform_gradient,
    transform_hessian,
    shape_function,
    shape_function_gradient,
    shape_function_hessian_vectorized,
    shape_function_on_edge,
)


class Interpolation(ABC):
    """Abstract interpolation class."""

    def __init__(self, grid: Grid, params: list):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    @abstractmethod
    def __call__(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ):
        """Compute the value of the interpolation in a triangle."""

    @abstractmethod
    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates | None
    ) -> Vector:
        """Compute the gradient of the interpolation in a triangle."""

    # @abstractmethod
    # def on_edge(self, edge: Edge, t: float):
    #     """Computes the value of the interpolation on an edge."""

    # @abstractmethod
    # def hessian(
    #     self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    # ) -> Matrix:
    #     """Compute the hessian of the interpolation in a triangle."""


class Deformation(ABC):
    """Abstract deformation class"""

    def __init__(
        self, y1_interpolation: Interpolation, y2_interpolation: Interpolation
    ):
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
        barycentric_coordinates: BarycentricCoordinates | None = None,
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


class P1Interpolation(Interpolation):
    """Piecewise affine interpolation."""

    def __call__(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ):
        i1, i2, i3 = triangle
        l1, l2, l3 = barycentric_coordinates
        return l1 * self._params[i1] + l2 * self._params[i2] + l3 * self._params[i3]

    def gradient(self, triangle: Triangle, barycentric_coordinates=None) -> Vector:
        # Unused argument 'barycentric_coordinates' assures the same calling convention
        # as in the case of the C1 interpolation
        del barycentric_coordinates
        i1, i2, i3 = triangle
        barycentric_gradient = Vector(
            [self._params[i1], self._params[i2], self._params[i3]]
        )
        return transform_gradient(
            self._grid.triangle_vertices(triangle), barycentric_gradient
        )

    def on_edge(self, edge: Edge, t: float):
        """Computes the interpolation on an edge."""
        i1, i2 = edge
        return t * self._params[i1] + (1 - t) * self._params[i2]

    def hessian(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Matrix:
        raise NotImplementedError("Hessian cannot be computed for P1 interpolations")


class P1Deformation(Deformation):
    """Deformation computed via piecewise affine interpolation."""

    def __init__(self, grid, y1_params: list, y2_params: list):
        super().__init__(
            P1Interpolation(grid, y1_params), P1Interpolation(grid, y2_params)
        )


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

        params = [Vector(p) for p in params]

        super().__init__(grid, params)

    def _triangle_params(self, triangle: Triangle) -> Vector:
        i1, i2, i3 = triangle
        return self._params[i1].extend(self._params[i2]).extend(self._params[i3])

    def __call__(
        self,
        triangle: Triangle,
        barycentric_coordinates: BarycentricCoordinates,
    ):
        """Compute the value of the interpolation in a triangle."""
        return shape_function(
            self._grid.triangle_vertices(triangle), barycentric_coordinates
        ).dot(self._triangle_params(triangle))

    def gradient(
        self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
    ) -> Vector:
        """Computes gradient of the interpolation."""
        triangle_vertices = self._grid.triangle_vertices(triangle)
        triangle_params = self._triangle_params(triangle)
        barycentric_gradient = (
            shape_function_gradient(triangle_vertices, barycentric_coordinates)
            .transpose()
            .dot(triangle_params)
        )
        return transform_gradient(triangle_vertices, barycentric_gradient)


# class C1Deformation:
#     """Deformation computed via C1 interpolation."""

#     def __init__(self, grid, y1_params: list[list], y2_params: list[list]):
#         self._y = [C1Interpolation(grid, y1_params), C1Interpolation(grid, y2_params)]

#     def __call__(
#         self,
#         triangle: Triangle,
#         barycentric_coordinates: BarycentricCoordinates,
#     ) -> Vector:
#         return Vector(
#             [
#                 self._y[0](triangle, barycentric_coordinates),
#                 self._y[1](triangle, barycentric_coordinates),
#             ]
#         )

#     def strain(
#         self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
#     ) -> Matrix:
#         """Computes strain of the deformation."""
#         return (
#             self._y[0]
#             .gradient(triangle, barycentric_coordinates)
#             .stack(self._y[1].gradient(triangle, barycentric_coordinates))
#         )

#     def hyper_strain(
#         self, triangle: Triangle, barycentric_coordinates: BarycentricCoordinates
#     ) -> Tensor3D:
#         """Computes the hyper strain of the deformation."""
#         hessian_y1 = self._y[0].hessian(triangle, barycentric_coordinates)
#         hessian_y2 = self._y[1].hessian(triangle, barycentric_coordinates)
#         return hessian_y1.stack(hessian_y2)

#     def on_edge(self, edge: Edge, t: float) -> Vector:
#         """Computes deformation on an edge."""
#         return Vector([self._y[0].on_edge(edge, t), self._y[1].on_edge(edge, t)])
