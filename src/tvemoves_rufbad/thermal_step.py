"""Module providing implementation of the thermal step of the minimizing movement scheme."""

from dataclasses import dataclass
from typing import Protocol
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.domain import Grid, RefinedGrid


class AbstractThermalStep(Protocol):
    """Abstract base class of a mechanical step returned by the mechanical_step factory."""

    def solve(self) -> None:
        """Solve the next mechanical step."""

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as numpy array."""

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as numpy array."""

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as numpy array."""

    def theta(self) -> npt.NDArray[np.float64]:
        """Return the current temperature as numpy array."""


@dataclass
class ThermalStepParams:
    """Parameters needed to perform the mechanical step."""

    search_radius: float
    shape_memory_scaling: float
    fps: int
    regularization: float


def _model(
    grid: Grid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    search_radius: float,
    shape_memory_scaling: float,
    fps: int,
) -> pyo.ConcreteModel:
    """Create model for the thermal step without regularization."""
    m = pyo.ConcreteModel("Thermal Step")

    if prev_y.shape != (len(grid.vertices), 2):
        raise ValueError(
            f"prev_y has incorrect shape {prev_y} (should be {(len(grid.vertices), 2)}"
        )
    if y.shape != (len(grid.vertices), 2):
        raise ValueError(f"y has incorrect shape {y} (should be {(len(grid.vertices), 2)}")

    if prev_theta.shape != (len(grid.vertices),):
        raise ValueError(f"y has incorrect shape {prev_theta} (should be {(len(grid.vertices),)}")

    m.prev_y1 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=prev_y[:, 0],
        mutable=True,
    )
    m.prev_y2 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=prev_y[:, 1],
        mutable=True,
    )
    m.y1 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=y[:, 0],
        mutable=True,
    )
    m.y2 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=y[:, 1],
        mutable=True,
    )
    m.prev_theta = pyo.Param(
        grid.vertices,
        within=pyo.NonNegativeReals,
        initialize=prev_theta,
        mutable=True,
    )

    # m.y1 = pyo.Var(grid.vertices, within=pyo.Reals)
    # m.y2 = pyo.Var(grid.vertices, within=pyo.Reals)
    # for v in grid.vertices:
    #     m.y1[v] = m.prev_y1[v]
    #     m.y1[v].bounds = (
    #         m.prev_y1[v] - search_radius,
    #         m.prev_y1[v] + search_radius,
    #     )
    #     m.y2[v] = m.prev_y2[v]
    #     m.y2[v].bounds = (
    #         m.prev_y2[v] - search_radius,
    #         m.prev_y2[v] + search_radius,
    #     )

    # for v in grid.dirichlet_boundary.vertices:
    #     m.y1[v].fix()
    #     m.y2[v].fix()

    # prev_y1 = P1Interpolation(grid, m.prev_y1)
    # prev_y2 = P1Interpolation(grid, m.prev_y2)
    # prev_deform = Deformation(prev_y1, prev_y2)

    # prev_temp = P1Interpolation(grid, m.prev_theta)

    # y1 = P1Interpolation(grid, m.y1)
    # y2 = P1Interpolation(grid, m.y2)
    # deform = Deformation(y1, y2)

    # integrator = Integrator(DUNAVANT2, grid.triangles, grid.points)
    # integrator_for_piecewise_constant = Integrator(CENTROID, grid.triangles, grid.points)

    # m.total_elastic_energy = integrator(
    #     _total_elastic_integrand(shape_memory_scaling, deform.strain, prev_temp)
    # )
    # m.dissipation = integrator_for_piecewise_constant(
    #     compose_to_integrand(dissipation_potential, prev_deform.strain, deform.strain)
    # )
    # m.objective = pyo.Objective(expr=m.total_elastic_energy + fps * m.dissipation)

    return m


class _ThermalStep(AbstractThermalStep):
    """Thermal step without regularization using P1 finite elements for the deformation."""

    def __init__(
        self,
        solver,
        grid: Grid,
        prev_y: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        prev_theta: npt.NDArray[np.float64],
        search_radius: float,
        shape_memory_scaling: float,
        fps: int,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._model = _model(
            grid,
            prev_y,
            y,
            prev_theta,
            search_radius,
            shape_memory_scaling,
            fps,
        )

    # def solve(self) -> None:
    #     self._solver.solve(self._model)

    # def prev_y(self) -> npt.NDArray[np.float64]:
    #     """Return the previous deformation as Nx2 numpy array, where N is the number of vertices."""
    #     return np.array(
    #         [
    #             [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
    #             for i in range(self._num_vertices)
    #         ]
    #     )

    # def y(self) -> npt.NDArray[np.float64]:
    #     """Return the current deformation as Nx2 numpy array, where N is the number of vertices."""
    #     return np.array(
    #         [[self._model.y1[i].value, self._model.y2[i].value] for i in range(self._num_vertices)]
    #     )

    # def prev_theta(self) -> npt.NDArray[np.float64]:
    #     """Return the previous temperature as vector of length N, where N is the number of vertices."""
    #     return np.array([self._model.prev_theta[i].value for i in range(self._num_vertices)])


def thermal_step(
    solver,
    grid: Grid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    params: ThermalStepParams,
    refined_grid: RefinedGrid | None = None,
) -> AbstractThermalStep:
    """Mechanical step factory."""
    if params.regularization == 0.0:
        return _ThermalStep(
            solver,
            grid,
            prev_y,
            y,
            prev_theta,
            params.search_radius,
            params.shape_memory_scaling,
            params.fps,
        )

    if refined_grid is None:
        raise ValueError("A refined grid must be provided for a regularized model")

    print("Regularized thermal step needs to be implemented")
