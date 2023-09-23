from math import isclose
from numpy.polynomial.legendre import leggauss


class TriangleQuadratureRule:
    def __init__(self, points, weights):
        for p in points:
            x, y = p
            if x < 0 or y < 0 or x + y > 1:
                raise ValueError(
                    "quadrature points not all contained in an equalteral unit triangle"
                )
        if not isclose(sum(w for w in weights), 1):
            raise ValueError("weights must sum up to 1")

        self.points = points
        self.weights = weights


CENTROID = TriangleQuadratureRule([(1 / 3, 1 / 3)], [1])
VERTEX = TriangleQuadratureRule([(1, 0), (0, 1), (0, 0)], [1 / 3, 1 / 3, 1 / 3])
DUNAVANT2 = TriangleQuadratureRule(
    [(2 / 3, 1 / 6), (1 / 6, 2 / 3), (1 / 6, 1 / 6)], [1 / 3, 1 / 3, 1 / 3]
)
DUNAVANT3 = TriangleQuadratureRule(
    [(1 / 3, 1 / 3), (0.6, 0.2), (0.2, 0.6), (0.2, 0.2)],
    [-0.5 - 0.625, 0.5 + 0.625 / 3, 0.5 + 0.625 / 3, 0.5 + 0.625 / 3],
)
DUNAVANT4 = TriangleQuadratureRule(
    [
        (0.108103018168070, 0.445948490915965),
        (0.445948490915965, 0.108103018168070),
        (0.108103018168070, 0.108103018168070),
        (0.091576213509771, 0.816847572980459),
        (0.816847572980459, 0.091576213509771),
        (0.091576213509771, 0.091576213509771),
    ],
    [
        0.223381589678011,
        0.223381589678011,
        0.223381589678011,
        0.109951743655322,
        0.109951743655322,
        0.109951743655322,
    ],
)
DUNAVANT5 = TriangleQuadratureRule(
    [
        (1 / 3, 1 / 3),
        (0.059715871789770, 0.470142064105115),
        (0.470142064105115, 0.059715871789770),
        (0.059715871789770, 0.059715871789770),
        (0.101286507323456, 0.797426985353087),
        (0.797426985353087, 0.101286507323456),
        (0.101286507323456, 0.101286507323456),
    ],
    [
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ],
)


class GaussQuadratureRule:
    def __init__(self, degree):
        # points are in the interval [-1, 1]
        points, weights = leggauss(degree)
        # transform the points and weights to the unit interval [0, 1]
        points = (1 + points) / 2
        weights /= 2
        self.points = list(points)
        self.weights = list(weights)
