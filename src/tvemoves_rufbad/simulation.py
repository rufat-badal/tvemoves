import pyomo.environ as pyo
from .grid import Grid
from .mechanical_step import MechanicalStepRegularized
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np


@dataclass(frozen=True)
class Step:
    y: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]

    def __repr__(self):
        return f"Step(y={repr(self.y)}, theta={repr(self.theta)})"

    def __str__(self):
        return f"y:\n{str(self.y)}\ntheta:\n{str(self.theta)}"


class Simulation:
    def __init__(
        self,
        grid: Grid,
        initial_temperature: float,
        search_radius: float,
        shape_memory_scaling: float,
        fps: int,
    ):
        self._solver = pyo.SolverFactory("ipopt")
        self._mechanical_step = MechanicalStepRegularized(
            self._solver,
            grid,
            initial_temperature,
            search_radius,
            shape_memory_scaling,
            fps,
        )
        # self._mechanical_step.solve()
        # self.steps = [
        #     Step(self._mechanical_step.prev_y(), self._mechanical_step.prev_theta())
        # ]
