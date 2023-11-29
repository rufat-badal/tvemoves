"""Module providing quadrature rules in triangles and on edges."""

from math import isclose
from typing import Iterator
from numpy.polynomial.legendre import leggauss


class TriangleQuadratureRule:
    """Quadrature rule inside a triangle."""

    def __init__(self, weights: list[float], points: list[tuple[float, float]]):
        for p in points:
            x, y = p
            if x < 0 or y < 0 or x + y > 1:
                raise ValueError(
                    "quadrature points not all contained in an equalteral unit triangle"
                )
        if not isclose(sum(w for w in weights), 1):
            raise ValueError("weights must sum up to 1")

        self.points = [(p[0], p[1], 1 - p[0] - p[1]) for p in points]
        self.weights = weights

    def __iter__(self) -> Iterator[tuple[float, tuple[float, float, float]]]:
        return zip(self.weights, self.points)


CENTROID = TriangleQuadratureRule([1], [(1 / 3, 1 / 3)])
VERTEX = TriangleQuadratureRule([1 / 3, 1 / 3, 1 / 3], [(1, 0), (0, 1), (0, 0)])

# see https://doi.org/10.1002/nme.1620210612

DUNAVANT2 = TriangleQuadratureRule(
    [1 / 3, 1 / 3, 1 / 3], [(2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 6, 1 / 6)]
)
DUNAVANT3 = TriangleQuadratureRule(
    [-0.5 - 0.0625, 0.5 + 0.0625 / 3, 0.5 + 0.0625 / 3, 0.5 + 0.0625 / 3],
    [(1 / 3, 1 / 3), (0.6, 0.2), (0.2, 0.6), (0.2, 0.2)],
)
DUNAVANT4 = TriangleQuadratureRule(
    [
        0.223381589678011,
        0.223381589678011,
        0.223381589678011,
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
    ],
    [
        (0.108103018168070, 0.445948490915965),
        (0.445948490915965, 0.108103018168070),
        (0.445948490915965, 0.445948490915965),
        (0.816847572980459, 0.091576213509771),
        (0.091576213509771, 0.816847572980459),
        (0.091576213509771, 0.091576213509771),
    ],
)
# This rule suffices to integrate Bell finite elements precisely
DUNAVANT5 = TriangleQuadratureRule(
    [
        0.225,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
    [
        (1 / 3, 1 / 3),
        (0.059715871789770, 0.470142064105115),
        (0.470142064105115, 0.059715871789770),
        (0.470142064105115, 0.470142064105115),
        (0.797426985353087, 0.101286507323456),
        (0.101286507323456, 0.797426985353087),
        (0.101286507323456, 0.101286507323456),
    ],
)

# CENTROID coincides with DUNAVANT1
DUNAVANT_QUADRATURE_RULES = [
    CENTROID,
    DUNAVANT2,
    DUNAVANT3,
    DUNAVANT4,
    DUNAVANT5,
]


class GaussQuadratureRule:
    """Gauss quadrature rule on the unit segment."""

    def __init__(self, degree: int):
        # points are in the interval [-1, 1]
        points, weights = leggauss(degree)
        # transform the points and weights to the unit interval [0, 1]
        points = (1 + points) / 2
        weights /= 2
        self.points = list(points)
        self.weights = list(weights)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return zip(self.weights, self.points)
