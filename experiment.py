from matplotlib import pyplot as plt
from tvemoves_rufbad.grid import SquareEquilateralGrid


grid = SquareEquilateralGrid(num_horizontal_points=5)
grid.plot()
plt.show()
