import pyomo.environ as pyo
from tvemoves_rufbad.grid import Grid
from tvemoves_rufbad.mechanical_step import MechanicalStepRegularized
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from tvemoves_rufbad.interpolation import P1Deformation


class Step:
    def __init__(
        self,
        y_params: npt.NDArray[np.float64],
        theta_params: npt.NDArray[np.float64],
        grid: Grid,
    ):
        self._y_params = y_params
        self._theta_params = theta_params

        self.y = P1Deformation(grid, *y_params)
        self.theta = P1Interpolation(grid, theta_params)

    def __repr__(self):
        return f"Step(y={repr(self._y_params)}, theta={repr(self._theta_params)})"

    def __str__(self):
        return f"y:\n{str(self._y_params)}\ntheta:\n{str(self._theta_params)}"


class Simulation:
    def __init__(
        self,
        grid: Grid,
        initial_temperature: float,
        search_radius: float,
        shape_memory_scaling: float,
        fps: int,
    ):
        self._grid = grid
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
        #     Step(
        #         self._mechanical_step.prev_y(),
        #         self._mechanical_step.prev_theta(),
        #         grid,
        #     )
        # ]
