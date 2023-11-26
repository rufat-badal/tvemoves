"""Test simulation class."""

from tvemoves_rufbad.grid import SquareEquilateralGrid
from tvemoves_rufbad.simulation import Simulation, SimulationParams


def test_simulation() -> None:
    """Assure that a simulation class can be correctly initialized."""
    grid = SquareEquilateralGrid(num_horizontal_points=2, fix="left")
    params = SimulationParams(
        initial_temperature=0.1,
        search_radius=10,
        shape_memory_scaling=2,
        fps=3,
        regularization=None,
    )
    simulation = Simulation(grid, params)
