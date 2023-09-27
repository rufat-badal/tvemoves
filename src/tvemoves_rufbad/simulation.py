import pyomo.environ as pyo
from .mechanical_step import MechanicalStep


class Step:
    def __init__(self, y, theta):
        self.y = y
        self.theta = theta

    def __repr__(self):
        return f"Step(y={repr(self.y)}, theta={repr(self.theta)})"

    def __str__(self):
        return f"y:\n{str(self.y)}\ntheta:\n{str(self.theta)}"


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
        self.steps = [
            Step(self._mechanical_step.prev_y(), self._mechanical_step.prev_theta())
        ]
        print(self.steps[-1])
