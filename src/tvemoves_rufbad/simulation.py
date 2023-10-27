"""Module provind the simulation class."""

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.grid import Grid
from tvemoves_rufbad.mechanical_step import (
    MechanicalStepParams,
    MechanicalStep,
)


@dataclass(frozen=True)
class Step:
    """Single simulation step (with or without regularization)"""

    y: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]

    def __repr__(self):
        return f"Step(y={repr(self.y)}, theta={repr(self.theta)})"

    def __str__(self):
        return f"y:\n{str(self.y)}\ntheta:\n{str(self.theta)}"


@dataclass
class SimulationParams:
    """Parameters specifying the simulation."""

    initial_temperature: float
    search_radius: float
    shape_memory_scaling: float
    fps: int
    regularization: float | None

    def get_mechanical_step_params(
        self,
    ) -> MechanicalStepParams:
        """Return subset of the parameters concerning the mechanical step only."""

        return MechanicalStepParams(
            self.initial_temperature,
            self.search_radius,
            self.shape_memory_scaling,
            self.fps,
            self.regularization,
        )


class Simulation:
    """Class implementing the minimizing movement scheme with or without regularization."""

    def __init__(self, grid: Grid, params: SimulationParams):
        self._grid = grid
        self._solver = pyo.SolverFactory("ipopt")
        self.params = params
        if params.regularization is None:
            self._mechanical_step = MechanicalStep(
                self._solver, grid, self.params.get_mechanical_step_params()
            )
        self._mechanical_step.solve()
        self.steps = [
            Step(self._mechanical_step.prev_y(), self._mechanical_step.prev_theta())
        ]
