from tvemoves_rufbad.grid import generate_square_equilateral_grid
from tvemoves_rufbad.simulation import Simulation

grid = generate_square_equilateral_grid(num_horizontal_points=3, fix="left")
simulation = Simulation(
    grid, initial_temperature=0.1, search_radius=10, shape_memory_scaling=2, fps=3
)
