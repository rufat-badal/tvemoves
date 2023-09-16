import pyomo.environ as pyo
from grid import Grid


class MechanicalStep:
    def __init__(self, grid: Grid):
        self._model = pyo.ConcreteModel()
        self._grid = grid
        m = self._model
