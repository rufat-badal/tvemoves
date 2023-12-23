"""Module provind the simulation class."""

from dataclasses import dataclass
from abc import ABC
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.domain import Domain, RectangleDomain, Grid, RefinedGrid
from tvemoves_rufbad.mechanical_step import MechanicalStepParams, mechanical_step
from tvemoves_rufbad.interpolation import (
    Deformation,
    Interpolation,
    P1Interpolation,
    C1Interpolation,
    RefinedInterpolation,
)


class AbstractStep(ABC):
    """Abstract simulation step"""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_deta: npt.NDArray[np.float64],
        y: Deformation,
        theta: Interpolation,
    ):
        self._y_data = y_data
        self._theta_data = theta_deta
        self.y = y
        self.theta = theta

    def __repr__(self):
        return f"Step(y={repr(self._y_data)}, theta={repr(self._theta_data)})"

    def __str__(self):
        return f"y:\n{str(self._y_data)}\ntheta:\n{str(self._theta_data)}"


class Step(AbstractStep):
    """Not regularized simulation step."""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_data: npt.NDArray[np.float64],
        grid: Grid,
    ):
        if y_data.shape != (len(grid.vertices), 2):
            raise ValueError(f"incorrectly shaped y_data of shape = {y_data.shape} provided")

        if theta_data.shape != (len(grid.vertices),):
            raise ValueError(
                f"incorrectly shaped theta_data provided of shape = {theta_data.shape} provided"
            )

        y1_params = y_data[:, 0].tolist()
        y1_interpolation = P1Interpolation(grid, y1_params)
        y2_params = y_data[:, 1].tolist()
        y2_interpolation = P1Interpolation(grid, y2_params)
        y = Deformation(y1_interpolation, y2_interpolation)

        theta = P1Interpolation(grid, theta_data.tolist())

        super().__init__(y_data, theta_data, y, theta)


class RegularizedStep(AbstractStep):
    """Not regularized simulation step."""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_data: npt.NDArray[np.float64],
        grid: Grid,
        refined_grid: RefinedGrid,
    ):
        # last dimension correspond to the degrees of freedom of the C1 interpolation
        if y_data.shape != (len(grid.vertices), 2, 6):
            raise ValueError(f"incorrectly shaped y_data of shape = {y_data.shape} provided")

        if theta_data.shape != (len(refined_grid.vertices),):
            raise ValueError(
                f"incorrectly shaped theta_data provided of shape = {theta_data.shape} provided"
            )

        y1_params = y_data[:, 0, :].tolist()
        y1_interpolation = RefinedInterpolation(C1Interpolation(grid, y1_params), refined_grid)
        y2_params = y_data[:, 1, :].tolist()
        y2_interpolation = RefinedInterpolation(C1Interpolation(grid, y2_params), refined_grid)
        y = Deformation(y1_interpolation, y2_interpolation)

        theta = P1Interpolation(refined_grid, theta_data.tolist())

        super().__init__(y_data, theta_data, y, theta)


@dataclass
class SimulationParams:
    """Parameters specifying the simulation."""

    initial_temperature: float
    search_radius: float
    shape_memory_scaling: float
    fps: int
    scale: float
    regularization: float = 0.0
    refinement_factor: int = 1

    def __post_init__(self):
        if self.initial_temperature < 0:
            raise ValueError(
                f"initial temperature must be non-negative but {self.initial_temperature} was"
                " provided"
            )

        if self.search_radius <= 0:
            raise ValueError(
                f"search radius must be positive but {self.search_radius} was provided"
            )

        if self.fps <= 0:
            raise ValueError(f"fps must be positive but {self.fps} was provided")

        if self.regularization < 0:
            raise ValueError(
                f"regularization must be non-negative but {self.regularization} was provided"
            )

        if self.scale <= 0:
            raise ValueError(f"scale must be positive but {self.scale} was provided")

        if self.refinement_factor <= 0:
            raise ValueError(f"fps must be positive but {self.refinement_factor} was provided")

    def mechanical_step_params(
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

    def __init__(self, domain: Domain, params: SimulationParams):
        self._domain = domain
        self._solver = pyo.SolverFactory("ipopt")
        self.params = params
        self._grid = self._domain.grid(self.params.scale)
        self._refined_grid = None
        if self.params.regularization != 0:
            self._refined_grid = self._domain.refine(self._grid, self.params.refinement_factor)
        self.steps: list[Step] = []

        self._mechanical_step = mechanical_step(
            self._solver, self._grid, self.params.mechanical_step_params(), self._refined_grid
        )
        self._append_step(self._mechanical_step.prev_y(), self._mechanical_step.prev_theta())
        print(self.steps[-1])

    def _append_step(self, y_data: npt.NDArray[np.float64], theta_data: npt.NDArray[np.float64]):
        step = (
            Step(y_data, theta_data, self._grid)
            if self.params.regularization == 0
            else RegularizedStep(y_data, theta_data, self._grid, self._refined_grid)
        )
        self.steps.append(step)


_params = SimulationParams(
    initial_temperature=0.1,
    search_radius=10.0,
    shape_memory_scaling=2.0,
    fps=1,
    regularization=1.0,
    scale=1.0,
    refinement_factor=5,
)

_square = RectangleDomain(1, 1, fix="left")
_simulation = Simulation(_square, _params)
