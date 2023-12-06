from tvemoves_rufbad.interpolation import P1Interpolation
from helpers import random_polynomial_2d


def test_p1_interpolation() -> None:
    poly, poly_gradient = random_polynomial_2d(1, 1)
    print(poly(1, 1))


test_p1_interpolation()
