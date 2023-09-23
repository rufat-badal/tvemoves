from tvemoves_rufbad.integrators import Integrator
from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.quadrature_rules import CENTROID

grid = SquareEquilateralGrid(num_horizontal_points=10)
integral = Integrator(CENTROID, grid)
