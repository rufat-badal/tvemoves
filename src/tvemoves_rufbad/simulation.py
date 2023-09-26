import pyomo.environ as pyo
from .mechanical_step import MechanicalStep


class Simulation:
    def __init__(
        self, grid, initial_temperature, search_radius, shape_memory_scaling, fps
    ):
        self._solver = pyo.SolverFactory("ipopt")
        self._mechanical_step = MechanicalStep(
            self._solver,
            grid,
            initial_temperature,
            search_radius,
            shape_memory_scaling,
            fps,
        )
        self._mechanical_step.solve()
