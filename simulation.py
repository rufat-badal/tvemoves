from mechanical_step import MechanicalStep
from grid import Grid


class Simulation:
    def __init__(self, grid: Grid):
        self._grid = grid
        self._mechanical_step = MechanicalStep(grid)
