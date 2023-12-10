"""Module providing implementation of the mechanical step of the minimizing movement scheme."""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import pyomo.environ as pyo
import numpy.typing as npt
import numpy as np
from tvemoves_rufbad.interpolation import p1_deformation, P1Interpolation, c1_deformation
from tvemoves_rufbad.quadrature_rules import CENTROID, DUNAVANT2, DUNAVANT5
from tvemoves_rufbad.tensors import Matrix
from tvemoves_rufbad.integrators import Integrator
from tvemoves_rufbad.domain import Grid
from tvemoves_rufbad.helpers import (
    create_martensite_potential,
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
    martensite_potential = create_martensite_potential(scaling_matrix)

    def total_elastic_potential(strain, theta):
        return austenite_percentage(theta) * austenite_potential(strain) + (
            1 - austenite_percentage(theta)
        ) * martensite_potential(strain)

    return compose_to_integrand(total_elastic_potential, strain, prev_temp)


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
    integrator_for_piecewise_constant = Integrator(CENTROID, grid.triangles, grid.points)

    prev_deform = p1_deformation(grid, m.prev_y1, m.prev_y2)
    deform = p1_deformation(grid, m.y1, m.y2)
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
        return np.array([
            [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
            for i in range(self._num_vertices)
        ])

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as Nx2 numpy array, where N is the number of vertices."""
        return np.array([
            [self._model.y1[i].value, self._model.y2[i].value] for i in range(self._num_vertices)
        ])

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as vector of length N, where N is the number of vertices."""
        return np.array([self._model.prev_theta[i].value for i in range(self._num_vertices)])


def _model_regularized(
    grid: Grid,
    initial_temperature: float,
    search_radius: float,
    shape_memory_scaling: float,
    fps: int,
    regularization: float,
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel("Mechanical Step with Regularization")
    m.vertices = pyo.RangeSet(len(grid.vertices))
    m.c1_indices = pyo.RangeSet(6)
    m.dirichlet_edges = pyo.RangeSet(len(grid.dirichlet_boundary.edges))
    m.deformation_indices = m.vertices * m.c1_indices
    m.dirichlet_constraint_indices = m.dirichlet_edges * m.c1_indices

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

    # Add dirichlet constraints.
    # We use the fact that y should be the identity map on any Dirichlet edge
    # and that C1 params encode all partial derivatives up to second order.
    # This way we can formulate for each component of y 3 conditions per edge vertex:
    # one for the value of the restriction of y to the edge, on for its first derivative,
    # and one for its second derivative.
    m.y1_constraints = pyo.Constraint(
        m.dirichlet_constraint_indices,
        rule=lambda model, i, j: _dirichlet_constraint_regularized(
            model.y1,
            grid.dirichlet_boundary.edges[i - 1],
            grid.edge_vertices(grid.dirichlet_boundary.edges[i - 1]),
            constraint_id=j,
            y_component=0,
        ),
    )
    m.y2_constraints = pyo.Constraint(
        m.dirichlet_constraint_indices,
        rule=lambda model, i, j: _dirichlet_constraint_regularized(
            model.y2,
            grid.dirichlet_boundary.edges[i - 1],
            grid.edge_vertices(grid.dirichlet_boundary.edges[i - 1]),
            constraint_id=j,
            y_component=1,
        ),
    )

    integrator = Integrator(DUNAVANT5, grid.triangles, grid.points)

    prev_y1_params = [[m.prev_y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_y2_params = [[m.prev_y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y1_params = [[m.y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y2_params = [[m.y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_deform = c1_deformation(grid, prev_y1_params, prev_y2_params)
    deform = c1_deformation(grid, y1_params, y2_params)

    return m


def _dirichlet_constraint_regularized(y, edge, edge_vertices, constraint_id, y_component):
    p, q = edge_vertices
    d = q - p
    i, j = edge
    i += 1
    j += 1
    match constraint_id:
        case 1:
            return y[i, 1] == p[y_component]
        case 2:
            return y[i, 2] * d[0] + y[i, 3] * d[1] == d[y_component]
        case 3:
            return y[i, 4] * d[0] ** 2 + 2 * y[i, 5] * d[0] * d[1] + y[i, 6] * d[1] ** 2 == 0
        case 4:
            return y[j, 1] == q[y_component]
        case 5:
            return y[j, 2] * d[0] + y[j, 3] * d[1] == d[y_component]
        case 6:
            return y[j, 4] * d[0] ** 2 + 2 * y[j, 5] * d[0] * d[1] + y[j, 6] * d[1] ** 2 == 0


class _MechanicalStepRegularized(AbstractMechanicalStep):
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
        self._model = _model_regularized(
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
        """Return the previous deformation as a Nx2x6 numpy array, where N is the number of vertices."""
        return np.array([
            [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
            for i in range(self._num_vertices)
        ])

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as a Nx2x6 numpy array, where N is the number of vertices."""
        return np.array([
            [self._model.y1[i].value, self._model.y2[i].value] for i in range(self._num_vertices)
        ])

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as a vector of length N, where N is the number of vertices."""
        return np.array([self._model.prev_theta[i].value for i in range(self._num_vertices)])


def mechanical_step(solver, grid: Grid, params: MechanicalStepParams) -> AbstractMechanicalStep:
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

    return _MechanicalStepRegularized(
        solver,
        grid,
        params.initial_temperature,
        params.search_radius,
        params.shape_memory_scaling,
        params.fps,
        params.regularization,
    )


_params = MechanicalStepParams(
    initial_temperature=0, search_radius=10, shape_memory_scaling=2, fps=3, regularization=1
)
_solver = pyo.SolverFactory("ipopt")
from tvemoves_rufbad.domain import RectangleDomain

_square = RectangleDomain(1, 1, fix="left")
_grid = _square.grid(1)
_mech_step = mechanical_step(_solver, _grid, _params)
# _mech_step._model.display()
