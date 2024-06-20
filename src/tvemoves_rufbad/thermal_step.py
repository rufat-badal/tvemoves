"""Module providing implementation of the thermal step of the minimizing movement scheme."""

from dataclasses import dataclass
from typing import Protocol, Callable
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from tvemoves_rufbad.domain import Grid, RefinedGrid
from tvemoves_rufbad.interpolation import (
    P1Interpolation,
    RefinedInterpolation,
    C1Interpolation,
    Deformation,
    Interpolation,
)
from tvemoves_rufbad.integrators import Integrator, BoundaryIntegrator
from tvemoves_rufbad.quadrature_rules import DUNAVANT2, DUNAVANT5
from tvemoves_rufbad.helpers import (
    heat_conductivity_reference,
    dissipation_rate,
    strain_derivative_coupling_potential,
    compose_to_integrand,
    ENTROPY_CONSTANT,
    HEAT_TRANSFER_COEFFICIENT,
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

    def update_prev_y(self, new_prev_y: npt.NDArray[np.float64]) -> None:
        """Update the previous deformation of the model.

        new_prev_y should be of the same format as returned by self.prev_y().
        """

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        """Update the previous temperature of the model. Automatically also updates the starting
        guess for theta and changes the search range.

        new_prev_theta should be of the same format as returned by self.prev_theta().
        """

    def update_y(self, new_y: npt.NDArray[np.float64]) -> None:
        """Update the current deformation of the model.

        new_y should be of the same format as returned by self.y().
        """

    def update_external_temperature(self, current_time: float) -> None:
        """Update the external temperature."""


@dataclass
class ThermalStepParams:
    """Parameters needed to perform the mechanical step."""

    search_radius: float
    fps: float
    regularization: float | None


def _energy_potential(fps: float):
    def energy_potential(prev_strain, strain, prev_temp, temp, temp_gradient):
        diffusion = temp_gradient.dot(heat_conductivity_reference(prev_strain).dot(temp_gradient))

        strain_rate = (strain - prev_strain) * fps
        heat_source_sink = (
            -(
                dissipation_rate(prev_strain, strain, fps)
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


def _boundary_potential(heat_transfer_coefficient: float, external_temp: float):
    def boundary_potential(temp):
        return heat_transfer_coefficient / 2 * (temp - external_temp) ** 2

    return boundary_potential


def _add_objective(
    m: pyo.ConcreteModel,
    integrator: Integrator,
    prev_deform: Deformation,
    deform: Deformation,
    prev_temp: Interpolation,
    temp: Interpolation,
    fps: float,
    boundary_integrator: BoundaryIntegrator | None = None,
):
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

    m.boundary_heat_transfer = 0.0
    if boundary_integrator is not None:
        m.boundary_heat_transfer = boundary_integrator(
            compose_to_integrand(
                _boundary_potential(HEAT_TRANSFER_COEFFICIENT, m.external_temperature), temp.on_edge
            )
        )

    m.objective = pyo.Objective(expr=m.energy + m.boundary_heat_transfer + fps * m.dissipation)


def _check_prev_step_data(
    y_data_shape: tuple[int, ...],
    theta_data_shape: tuple[int, ...],
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
):
    if prev_y.shape != y_data_shape:
        raise ValueError(f"prev_y has incorrect shape {prev_y.shape} (should be {y_data_shape}")
    if y.shape != y_data_shape:
        raise ValueError(f"y has incorrect shape {y.shape} (should be {y_data_shape}")
    if prev_theta.shape != theta_data_shape:
        raise ValueError(f"y has incorrect shape {prev_theta.shape} (should be {theta_data_shape}")


def _model(
    grid: Grid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    search_radius: float,
    fps: float,
    external_temperature: float,
) -> pyo.ConcreteModel:
    """Create model for the thermal step without regularization."""
    m = pyo.ConcreteModel("Thermal Step")

    _check_prev_step_data((len(grid.vertices), 2), (len(grid.vertices),), prev_y, y, prev_theta)

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
    m.external_temperature = pyo.Param(
        within=pyo.NonNegativeReals, initialize=external_temperature, mutable=True
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

    boundary_integrator = BoundaryIntegrator(1, grid.boundary.edges, grid.points)

    _add_objective(
        m,
        integrator,
        prev_deform,
        deform,
        prev_temp,
        temp,
        fps,
        boundary_integrator,
    )

    return m


def _model_regularized(
    grid: Grid,
    refined_grid: RefinedGrid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    search_radius: float,
    fps: float,
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel("Thermal Step with Regularization")

    _check_prev_step_data(
        (len(grid.vertices), 2, 6), (len(refined_grid.vertices),), prev_y, y, prev_theta
    )

    m.vertices = pyo.RangeSet(len(grid.vertices))
    m.c1_indices = pyo.RangeSet(6)
    m.deformation_indices = m.vertices * m.c1_indices
    m.refined_vertices = pyo.RangeSet(len(refined_grid.vertices))

    m.prev_y1 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: prev_y[i - 1, 0, j - 1],
        mutable=True,
    )
    m.prev_y2 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: prev_y[i - 1, 1, j - 1],
        mutable=True,
    )
    m.y1 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: y[i - 1, 0, j - 1],
        mutable=True,
    )
    m.y2 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: y[i - 1, 1, j - 1],
        mutable=True,
    )
    m.prev_theta = pyo.Param(
        m.refined_vertices,
        within=pyo.NonNegativeReals,
        initialize=lambda model, i: prev_theta[i - 1],
        mutable=True,
    )

    m.theta = pyo.Var(m.refined_vertices, within=pyo.NonNegativeReals)

    for i in list(m.refined_vertices):
        m.theta[i] = m.prev_theta[i]
        m.theta[i].bounds = (
            m.prev_theta[i] - search_radius,
            m.prev_theta[i] + search_radius,
        )

    prev_y1_params = [[m.prev_y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_y1 = RefinedInterpolation(C1Interpolation(grid, prev_y1_params), refined_grid)
    prev_y2_params = [[m.prev_y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_y2 = RefinedInterpolation(C1Interpolation(grid, prev_y2_params), refined_grid)
    prev_deform = Deformation(prev_y1, prev_y2)

    y1_params = [[m.y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y1 = RefinedInterpolation(C1Interpolation(grid, y1_params), refined_grid)
    y2_params = [[m.y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y2 = RefinedInterpolation(C1Interpolation(grid, y2_params), refined_grid)
    deform = Deformation(y1, y2)

    prev_temp = P1Interpolation(refined_grid, [m.prev_theta[i] for i in list(m.refined_vertices)])

    temp = P1Interpolation(refined_grid, [m.theta[i] for i in list(m.refined_vertices)])

    integrator = Integrator(DUNAVANT5, refined_grid.triangles, refined_grid.points)

    _add_objective(m, integrator, prev_deform, deform, prev_temp, temp, fps)

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
        external_temperature: Callable[[float], float],
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._model = _model(
            grid, prev_y, y, prev_theta, search_radius, fps, external_temperature(0)
        )
        self._search_radius = search_radius
        self._external_temperature = external_temperature

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

    def update_prev_y(self, new_prev_y: npt.NDArray[np.float64]) -> None:
        if new_prev_y.shape != (self._num_vertices, 2):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_y.shape} (should be"
                f" {(self._num_vertices, 2)})"
            )

        new_prev_y1 = new_prev_y[:, 0]
        new_prev_y2 = new_prev_y[:, 1]
        m = self._model
        for i in range(self._num_vertices):
            m.prev_y1[i] = new_prev_y1[i]
            m.prev_y2[i] = new_prev_y2[i]

    def update_y(self, new_y: npt.NDArray[np.float64]) -> None:
        if new_y.shape != (self._num_vertices, 2):
            raise ValueError(
                f"Input array has incorrect shape {new_y.shape} (should be"
                f" {(self._num_vertices, 2)})"
            )

        new_y1 = new_y[:, 0]
        new_y2 = new_y[:, 1]
        m = self._model
        for i in range(self._num_vertices):
            m.y1[i] = new_y1[i]
            m.prev_y2[i] = new_y2[i]

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        if new_prev_theta.shape != (self._num_vertices,):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_theta.shape} (should be"
                f" {(self._num_vertices,)})"
            )

        m = self._model
        for i in range(self._num_vertices):
            m.prev_theta[i] = new_prev_theta[i]
            m.theta[i] = new_prev_theta[i]
            m.theta[i].bounds = (
                new_prev_theta[i] - self._search_radius,
                new_prev_theta[i] + self._search_radius,
            )

    def update_external_temperate(self, current_time: float) -> None:
        self._model.external_temperature = self._external_temperature(current_time)


class _ThermalStepRegularized(AbstractThermalStep):
    """Thermal step without regularization using P1 finite elements for the deformation."""

    def __init__(
        self,
        solver,
        grid: Grid,
        refined_grid: RefinedGrid,
        prev_y: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        prev_theta: npt.NDArray[np.float64],
        search_radius: float,
        fps: float,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._search_radius = search_radius
        self._model = _model_regularized(
            grid,
            refined_grid,
            prev_y,
            y,
            prev_theta,
            search_radius,
            fps,
        )

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as a Nx2x6 numpy array, where N is the number of
        vertices."""
        m = self._model
        c1_indices = list(m.c1_indices)
        vertices = list(m.vertices)
        return np.array([
            [
                [m.prev_y1[i, j].value for j in c1_indices],
                [m.prev_y2[i, j].value for j in c1_indices],
            ]
            for i in vertices
        ])

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as a Nx2x6 numpy array, where N is the number of
        vertices."""
        m = self._model
        c1_indices = list(m.c1_indices)
        vertices = list(m.vertices)
        return np.array([
            [
                [m.y1[i, j].value for j in c1_indices],
                [m.y2[i, j].value for j in c1_indices],
            ]
            for i in vertices
        ])

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as a vector of length N_ref, where N_ref is the number of
        vertices of the refined grid.
        """
        return np.array(
            [self._model.prev_theta[i].value for i in list(self._model.refined_vertices)]
        )

    def theta(self) -> npt.NDArray[np.float64]:
        """Return the current temperature as a vector of length N_ref, where N_ref is the number of
        vertices of the refined grid.
        """
        return np.array([self._model.theta[i].value for i in list(self._model.refined_vertices)])

    def update_prev_y(self, new_prev_y: npt.NDArray[np.float64]) -> None:
        m = self._model
        c1_indices = list(m.c1_indices)
        vertices = list(m.vertices)

        if new_prev_y.shape != (len(vertices), 2, 6):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_y.shape} (should be"
                f" {(len(vertices), 2, 6)})"
            )

        new_prev_y1 = new_prev_y[:, 0, :]
        new_prev_y2 = new_prev_y[:, 1, :]
        for i in vertices:
            for j in c1_indices:
                m.prev_y1[i, j] = new_prev_y1[i - 1, j - 1]
                m.prev_y2[i, j] = new_prev_y2[i - 1, j - 1]

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        refined_vertices = list(self._model.refined_vertices)
        if new_prev_theta.shape != (len(refined_vertices),):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_theta.shape} (should be"
                f" {(len(refined_vertices),)})"
            )
        m = self._model
        for i in refined_vertices:
            m.prev_theta[i] = new_prev_theta[i - 1]
            m.theta[i] = new_prev_theta[i - 1]
            m.theta[i].bounds = (
                new_prev_theta[i - 1] - self._search_radius,
                new_prev_theta[i - 1] + self._search_radius,
            )

    def update_y(self, new_y: npt.NDArray[np.float64]) -> None:
        m = self._model
        c1_indices = list(m.c1_indices)
        vertices = list(m.vertices)

        if new_y.shape != (len(vertices), 2, 6):
            raise ValueError(
                f"Input array has incorrect shape {new_y.shape} (should be {(len(vertices), 2, 6)})"
            )

        new_y1 = new_y[:, 0, :]
        new_y2 = new_y[:, 1, :]
        for i in vertices:
            for j in c1_indices:
                m.y1[i, j] = new_y1[i - 1, j - 1]
                m.y2[i, j] = new_y2[i - 1, j - 1]


def thermal_step(
    solver,
    grid: Grid,
    prev_y: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prev_theta: npt.NDArray[np.float64],
    params: ThermalStepParams,
    external_temperature: Callable[[float], float],
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
            external_temperature,
        )

    if refined_grid is None:
        raise ValueError("A refined grid must be provided for a regularized model")

    return _ThermalStepRegularized(
        solver,
        grid,
        refined_grid,
        prev_y,
        y,
        prev_theta,
        params.search_radius,
        params.fps,
    )
