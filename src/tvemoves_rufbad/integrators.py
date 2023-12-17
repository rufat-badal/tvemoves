"""Module providing integrators over triangles or edges."""

from typing import Callable, Any
import pyomo.environ as pyo
from tvemoves_rufbad.quadrature_rules import GaussQuadratureRule, TriangleQuadratureRule
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.domain import Triangle, Edge, BarycentricCoordinates


class Integrator:
    """Integrator over triangles."""

    def __init__(
        self,
        quadrature: TriangleQuadratureRule,
        triangles: list[Triangle],
        points: list[Vector],
    ):
        self._triangles = list(triangles)
        self._quadrature = quadrature
        first_sides = (points[j] - points[i] for (i, j, _) in triangles)
        second_sides = (points[k] - points[i] for (i, _, k) in triangles)
        self._triangle_areas = [
            abs(first.stack(second).det() / 2) for (first, second) in zip(first_sides, second_sides)
        ]

    def __call__(
        self,
        integrand: Callable[[Triangle, BarycentricCoordinates], Any],
    ):
        return sum(
            triangle_area
            * sum(
                weight * integrand(triangle, point)
                for point, weight in zip(self._quadrature.points, self._quadrature.weights)
            )
            for (triangle_area, triangle) in zip(self._triangle_areas, self._triangles)
        )


class BoundaryIntegrator:
    """Integrator over edges."""

    def __init__(self, degree: int, edges: list[Edge], points: list[Vector]):
        self._edges = edges
        self._quadrature = GaussQuadratureRule(degree)
        self._edge_lengths = [pyo.sqrt((points[i] - points[j]).normsqr()) for (i, j) in edges]

    def __call__(
        self,
        integrand: Callable[[Edge, float], float],
    ):
        # integrand(edge, t) should return a float,
        # where edge is any edge in self._edges and t is in [0, 1]
        return sum(
            edge_length
            * sum(
                weight * integrand(edge, point)
                for (point, weight) in zip(self._quadrature.points, self._quadrature.weights)
            )
            for (edge_length, edge) in zip(self._edge_lengths, self._edges)
        )
