from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.simulation import Simulation

grid = SquareEquilateralGrid(num_horizontal_points=3, fix="left")
simulation = Simulation(
    grid, initial_temperature=0.1, search_radius=10, shape_memory_scaling=2, fps=3
)
simulation._mechanical_step._model.prev_y1.display()
simulation._mechanical_step._model.y1.display()
