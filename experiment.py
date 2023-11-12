from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.simulation import SimulationParams, Simulation

grid = SquareEquilateralGrid(num_horizontal_points=3)
params = SimulationParams(
    initial_temperature=0.0,
    search_radius=10.0,
    shape_memory_scaling=2.0,
    fps=3,
    regularization=None,
)
simulation = Simulation(grid, params)
print(simulation.steps[0].y)
