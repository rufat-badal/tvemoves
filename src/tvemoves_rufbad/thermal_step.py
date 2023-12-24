"""Module providing implementation of the thermal step of the minimizing movement scheme."""

from dataclasses import dataclass
from typing import Protocol
import numpy.typing as npt
import numpy as np
from tvemoves_rufbad.domain import Grid, RefinedGrid


class AbstractMechanicalStep(Protocol):
    """Abstract base class of a mechanical step returned by the mechanical_step factory."""

    def solve(self) -> None:
        """Solve the next mechanical step."""

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as numpy array."""

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as numpy array."""

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as numpy array."""


@dataclass
class ThermalStepParams:
    """Parameters needed to perform the mechanical step."""

    search_radius: float
    shape_memory_scaling: float
    fps: int
    regularization: float


def thermal_step(
    solver, grid: Grid, params: ThermalStepParams, refined_grid: RefinedGrid | None = None
) -> AbstractMechanicalStep:
    """Mechanical step factory."""
    if params.regularization == 0.0:
        print("Not regularized thermal step needs to be implemented")
        return

    if refined_grid is None:
        raise ValueError("A refined grid must be provided for a regularized model")

    print("Regularized thermal step needs to be implemented")
