from .quadrature_rules import GaussQuadratureRule
import pyomo.environ as pyo


class Integrator:
    def __init__(self, quadrature, triangles, points):
        self._triangles = triangles
        self._quadrature = quadrature
        first_sides = (points[j] - points[i] for (i, j, _) in self._triangles)
        second_sides = (points[k] - points[i] for (i, _, k) in self._triangles)
        self._triangle_areas = [
            abs(first.vstack(second).det() / 2)
            for (first, second) in zip(first_sides, second_sides)
        ]

    def __call__(self, integrand):
        # integrand(triangle, (t1, t2, t3)) should return a float,
        # where triangle is any triangle in self._triangles
        # and (t1, t2, t3) are barycentric coordinates
        return sum(
            triangle_area
            * sum(
                weight * integrand(triangle, point)
                for (point, weight) in zip(
                    self._quadrature.points, self._quadrature.weights
                )
            )
            for (triangle_area, triangle) in zip(self._triangle_areas, self._triangles)
        )


class BoundaryIntegrator:
    def __init__(self, degree, edges, points):
        self._edges = edges
        self._quadrature = GaussQuadratureRule(degree)
        self._edge_lengths = [
            pyo.sqrt((points[i] - points[j]).normsqr()) for (i, j) in self._edges
        ]

    def __call__(self, integrand):
        # integrand(edge, t) should return a float,
        # where edge is any edge in self._edges and t is in [0, 1]
        return sum(
            edge_length
            * sum(
                weight * integrand(edge, point)
                for (point, weight) in zip(
                    self._quadrature.points, self._quadrature.weights
                )
            )
            for (edge_length, edge) in zip(self._edge_lengths, self._edges)
        )
