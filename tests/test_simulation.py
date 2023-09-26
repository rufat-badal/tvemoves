from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.mechanical_step import MechanicalStep

grid = SquareEquilateralGrid(num_horizontal_points=2, fix="left")
step = MechanicalStep(
    grid, initial_temperature=1, search_radius=1, shape_memory_scaling=2, fps=3
)
step._model.display()
