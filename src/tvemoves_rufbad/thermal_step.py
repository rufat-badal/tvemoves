"""Module providing implementation of the thermal step of the minimizing movement scheme."""

from dataclasses import dataclass
from typing import Protocol
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.domain import Grid, RefinedGrid
from tvemoves_rufbad.interpolation import P1Interpolation, Deformation
from tvemoves_rufbad.integrators import Integrator
from tvemoves_rufbad.quadrature_rules import DUNAVANT2
from tvemoves_rufbad.helpers import (
    heat_conductivity_reference,
    dissipation_rate,
    strain_derivative_coupling_potential,
    compose_to_integrand,
    ENTROPY_CONSTANT,
    internal_energy_no_entropy,
    temp_antrider_internal_energy_no_entropy,
)


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
    fps: float
    regularization: float


def _energy_potential(fps: float):
    def energy_potential(prev_strain, strain, prev_temp, temp, temp_gradient):
        diffusion = temp_gradient.dot(heat_conductivity_reference(prev_strain).dot(temp_gradient))

        strain_rate = (strain - prev_strain) * fps
        heat_source_sink = (
            -(
                dissipation_rate(prev_strain, strain)
                + strain_derivative_coupling_potential(prev_strain, prev_temp).scalar_product(
                    strain_rate
                )
            )
            * temp
        )

        return diffusion + heat_source_sink

    return energy_potential


def _dissipation_potential(prev_strain, strain, prev_temp, temp):
    l2_dissipation = ENTROPY_CONSTANT / 2 * (temp - prev_temp) ** 2
    internal_energy_dissipation = (
        temp_antrider_internal_energy_no_entropy(strain, temp)
        - internal_energy_no_entropy(prev_strain, prev_temp) * temp
    )

    return l2_dissipation + internal_energy_dissipation


def _model(
    grid: Grid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    search_radius: float,
    fps: float,
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

    m.theta = pyo.Var(grid.vertices, within=pyo.NonNegativeReals)
    for v in grid.vertices:
        m.theta[v] = m.prev_theta[v]
        m.theta[v].bounds = (
            m.prev_theta[v] - search_radius,
            m.prev_theta[v] + search_radius,
        )

    prev_y1 = P1Interpolation(grid, m.prev_y1)
    prev_y2 = P1Interpolation(grid, m.prev_y2)
    prev_deform = Deformation(prev_y1, prev_y2)

    y1 = P1Interpolation(grid, m.y1)
    y2 = P1Interpolation(grid, m.y2)
    deform = Deformation(y1, y2)

    prev_temp = P1Interpolation(grid, m.prev_theta)

    temp = P1Interpolation(grid, m.theta)

    integrator = Integrator(DUNAVANT2, grid.triangles, grid.points)

    m.energy = integrator(
        compose_to_integrand(
            _energy_potential(fps),
            prev_deform.strain,
            deform.strain,
            prev_temp,
            temp,
            temp.gradient,
        )
    )

    m.dissipation = integrator(
        compose_to_integrand(
            _dissipation_potential,
            prev_deform.strain,
            deform.strain,
            prev_temp,
            temp,
        )
    )

    m.objective = pyo.Objective(expr=m.energy + fps * m.dissipation)

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
        fps: float,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._model = _model(grid, prev_y, y, prev_theta, search_radius, fps)

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as Nx2 numpy array, where N is the number of vertices."""
        return np.array([
            [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
            for i in range(self._num_vertices)
        ])

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as Nx2 numpy array, where N is the number of vertices."""
        return np.array(
            [[self._model.y1[i].value, self._model.y2[i].value] for i in range(self._num_vertices)]
        )

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as vector of length N, where N is the number of vertices."""
        return np.array([self._model.prev_theta[i].value for i in range(self._num_vertices)])

    def theta(self) -> npt.NDArray[np.float64]:
        """Return the current temperature as vector of length N, where N is the number of vertices."""
        return np.array([self._model.theta[i].value for i in range(self._num_vertices)])


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
    if params.regularization is None:
        return _ThermalStep(
            solver,
            grid,
            prev_y,
            y,
            prev_theta,
            params.search_radius,
            params.fps,
        )

    if refined_grid is None:
        raise ValueError("A refined grid must be provided for a regularized model")

    print("Regularized thermal step needs to be implemented")
