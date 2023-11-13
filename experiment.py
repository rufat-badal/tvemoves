from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.simulation import SimulationParams, Simulation

grid = SquareEquilateralGrid(num_horizontal_points=2)
params = SimulationParams(
    initial_temperature=0.0,
    search_radius=10.0,
    shape_memory_scaling=2.0,
    fps=3,
    regularization=1.0,
)
simulation = Simulation(grid, params)
simulation._mechanical_step._model.display()
# print(simulation.steps[0].y)
