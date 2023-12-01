"""Module providing implementation of the mechanical step of the minimizing movement scheme."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import pyomo.environ as pyo
import numpy.typing as npt
import numpy as np
from tvemoves_rufbad.interpolation import P1Deformation, P1Interpolation
from tvemoves_rufbad.quadrature_rules import DUNAVANT2, CENTROID
from tvemoves_rufbad.tensors import Matrix
from tvemoves_rufbad.integrators import Integrator
from tvemoves_rufbad.grid import Grid
from tvemoves_rufbad.utils import (
    martensite_potential,
    austenite_percentage,
    austenite_potential,
    compose_to_integrand,
    dissipation_potential,
)


@dataclass
class MechanicalStepParams:
    """Parameters needed to perform the mechanical step."""

    initial_temperature: float
    search_radius: float
    shape_memory_scaling: float
    fps: int
    regularization: float | None


class AbstractMechanicalStep(ABC):
    """Abstract base class of a mechanical step returned by the mechanical_step factory."""

    @abstractmethod
    def solve(self) -> None:
        """Solve the next mechanical step."""

    @abstractmethod
    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as numpy array."""

    @abstractmethod
    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as numpy array."""

    @abstractmethod
    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as numpy array."""


def _total_elastic_integrand(shape_memory_scaling, strain, prev_temp):
    """Construct integrand of the total elastic energy without regularizing terms."""

    scaling_matrix = Matrix([[1 / shape_memory_scaling, 0], [0, 1]])
    martensite_potential = martensite_potential(scaling_matrix)

    def martensite_percentage(theta):
        return 1 - austenite_percentage(theta)

    def total_elastic_potential(strain, theta):
        return austenite_percentage(theta) * austenite_potential(
            strain
        ) + martensite_percentage(theta) * martensite_potential(strain)

    total_elastic_integrand = compose_to_integrand(
        total_elastic_potential, strain, prev_temp
    )

    return total_elastic_integrand


def _model(
    grid: Grid,
    initial_temperature: float,
    search_radius: float,
    shape_memory_scaling: float,
    fps: int,
) -> pyo.ConcreteModel:
    """Create model for the mechanical step without regularization."""
    m = pyo.ConcreteModel("Mechanical Step")

    m.prev_y1 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=[p[0] for p in grid.points],
        mutable=True,
    )
    m.prev_y2 = pyo.Param(
        grid.vertices,
        within=pyo.Reals,
        initialize=[p[1] for p in grid.points],
        mutable=True,
    )
    m.prev_theta = pyo.Param(
        grid.vertices,
        within=pyo.NonNegativeReals,
        initialize=initial_temperature,
        mutable=True,
    )

    m.y1 = pyo.Var(grid.vertices, within=pyo.Reals)
    m.y2 = pyo.Var(grid.vertices, within=pyo.Reals)
    for v in grid.vertices:
        m.y1[v] = m.prev_y1[v]
        m.y1[v].bounds = (
            m.prev_y1[v] - search_radius,
            m.prev_y1[v] + search_radius,
        )
        m.y2[v] = m.prev_y2[v]
        m.y2[v].bounds = (
            m.prev_y2[v] - search_radius,
            m.prev_y2[v] + search_radius,
        )

    for v in grid.dirichlet_boundary.vertices:
        m.y1[v].fix()
        m.y2[v].fix()

    integrator = Integrator(DUNAVANT2, grid.triangles, grid.points)
    integrator_for_piecewise_constant = Integrator(
        CENTROID, grid.triangles, grid.points
    )

    prev_deform = P1Deformation(grid, m.prev_y1, m.prev_y2)
    deform = P1Deformation(grid, m.y1, m.y2)
    prev_temp = P1Interpolation(grid, m.prev_theta)

    m.total_elastic_energy = integrator(
        _total_elastic_integrand(shape_memory_scaling, deform.strain, prev_temp)
    )
    m.dissipation = integrator_for_piecewise_constant(
        compose_to_integrand(dissipation_potential, prev_deform.strain, deform.strain)
    )
    m.objective = pyo.Objective(expr=m.total_elastic_energy + fps * m.dissipation)

    return m


class _MechanicalStep(AbstractMechanicalStep):
    """Mechanical step without regularization using P1 finite elements for the deformation."""

    def __init__(
        self,
        solver,
        grid: Grid,
        initial_temperature: float,
        search_radius: float,
        shape_memory_scaling: float,
        fps: int,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._model = _model(
            grid,
            initial_temperature,
            search_radius,
            shape_memory_scaling,
            fps,
        )

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as Nx2 numpy array, where N is the number of vertices."""
        return np.array(
            [
                [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as Nx2 numpy array, where N is the number of vertices."""
        return np.array(
            [
                [self._model.y1[i].value, self._model.y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as vector of length N, where N is the number of vertices."""
        return np.array(
            [self._model.prev_theta[i].value for i in range(self._num_vertices)]
        )


def _model_regularized_bell_finite_elements(
    grid: Grid,
    initial_temperature: float,
    search_radius: float,
    shape_memory_scaling: float,
    fps: int,
    regularization: float,
) -> pyo.ConcreteModel:
    print("Hello from _model_regularized_bell_finite_elements")
    m = pyo.ConcreteModel("Mechanical Step with Regularization")
    m.vertices = pyo.RangeSet(len(grid.vertices))
    m.deformation_indices = m.vertices * pyo.RangeSet(6)

    initial_y1 = [[p[0], 1, 0, 0, 0, 0] for p in grid.points]
    initial_y2 = [[p[1], 0, 1, 0, 0, 0] for p in grid.points]

    m.prev_y1 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: initial_y1[i - 1][j - 1],
        mutable=True,
    )
    m.prev_y2 = pyo.Param(
        m.deformation_indices,
        within=pyo.Reals,
        initialize=lambda model, i, j: initial_y2[i - 1][j - 1],
        mutable=True,
    )

    m.y1 = pyo.Var(m.deformation_indices, within=pyo.Reals)
    m.y2 = pyo.Var(m.deformation_indices, within=pyo.Reals)

    for i in m.deformation_indices:
        m.y1[i] = m.prev_y1[i]
        m.y1[i].bounds = (
            m.prev_y1[i] - search_radius,
            m.prev_y1[i] + search_radius,
        )
        m.y2[i] = m.prev_y2[i]
        m.y2[i].bounds = (
            m.prev_y2[i] - search_radius,
            m.prev_y2[i] + search_radius,
        )

    for v in grid.dirichlet_boundary.vertices:
        m.y1[v + 1, 1].fix()
        m.y2[v + 1, 1].fix()

    return m


class _MechanicalStepRegularizedBellFiniteElements(AbstractMechanicalStep):
    """Mechanical step with regularization using Bell finite elements for the deformation."""

    def __init__(
        self,
        solver,
        grid: Grid,
        initial_temperature: float,
        search_radius: float,
        shape_memory_scaling: float,
        fps: int,
        regularization: float,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._model = _model_regularized_bell_finite_elements(
            grid,
            initial_temperature,
            search_radius,
            shape_memory_scaling,
            fps,
            regularization,
        )

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as Nx2x6 numpy array, where N is the number of vertices."""
        return np.array(
            [
                [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as Nx2x6 numpy array, where N is the number of vertices."""
        return np.array(
            [
                [self._model.y1[i].value, self._model.y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as vector of length N, where N is the number of vertices."""
        return np.array(
            [self._model.prev_theta[i].value for i in range(self._num_vertices)]
        )


def mechanical_step(
    solver, grid: Grid, params: MechanicalStepParams
) -> AbstractMechanicalStep:
    """Mechanical step factory."""
    if params.regularization is None:
        return _MechanicalStep(
            solver,
            grid,
            params.initial_temperature,
            params.search_radius,
            params.shape_memory_scaling,
            params.fps,
        )

    return _MechanicalStepRegularizedBellFiniteElements(
        solver,
        grid,
        params.initial_temperature,
        params.search_radius,
        params.shape_memory_scaling,
        params.fps,
        params.regularization,
    )
