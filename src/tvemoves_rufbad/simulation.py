"""Module provind the simulation class."""

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.domain import Domain, RectangleDomain
from tvemoves_rufbad.mechanical_step import MechanicalStepParams, mechanical_step
from tvemoves_rufbad.interpolation import Interpolation, Deformation


@dataclass(frozen=True)
class Step:
    """Single simulation step (with or without regularization)"""

    _y_data: npt.NDArray[np.float64]
    _theta_data: npt.NDArray[np.float64]

    def __repr__(self):
        return f"Step(y={repr(self._y_data)}, theta={repr(self._theta_data)})"

    def __str__(self):
        return f"y:\n{str(self._y_data)}\ntheta:\n{str(self._theta_data)}"


@dataclass
class SimulationParams:
    """Parameters specifying the simulation."""

    initial_temperature: float
    search_radius: float
    shape_memory_scaling: float
    fps: int
    regularization: float | None
    scale: float
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
        grid = self._domain.grid(self.params.scale)
        refined_grid = None
        if self.params.regularization != 0:
            refined_grid = self._domain.refine(grid, self.params.refinement_factor)
        self._mechanical_step = mechanical_step(
            self._solver, grid, self.params.mechanical_step_params(), refined_grid
        )
        self._mechanical_step.solve()


_params = SimulationParams(
    initial_temperature=0.1,
    search_radius=10,
    shape_memory_scaling=2,
    fps=1,
    regularization=1,
    scale=1,
    refinement_factor=5,
)

_square = RectangleDomain(1, 1, fix="left")
_simulation = Simulation(_square, _params)
