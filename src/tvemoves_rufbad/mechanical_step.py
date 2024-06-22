"""Module providing implementation of the mechanical step of the minimizing movement scheme."""

from dataclasses import dataclass
from typing import Protocol, Callable
import pyomo.environ as pyo
import numpy.typing as npt
import numpy as np
from tvemoves_rufbad.interpolation import (
    P1Interpolation,
    P1BoundaryInterpolation,
    C1Interpolation,
    RefinedInterpolation,
    Deformation,
)
from tvemoves_rufbad.quadrature_rules import CENTROID, DUNAVANT2, DUNAVANT5
from tvemoves_rufbad.integrators import Integrator, BoundaryIntegrator
from tvemoves_rufbad.tensors import Vector
from tvemoves_rufbad.domain import Grid, RefinedGrid
from tvemoves_rufbad.helpers import (
    total_elastic_potential,
    compose_to_integrand,
    dissipation_potential,
    hyper_elastic_potential,
)

HYPER_STRAIN_POWER = 4


@dataclass
class MechanicalStepParams:
    """Parameters needed to perform the mechanical step."""

    initial_temperature: float
    search_radius: float
    fps: float
    regularization: float | None


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

    def update_prev_y(self, new_prev_y: npt.NDArray[np.float64]) -> None:
        """Update the previous deformation of the model. Automatically also updates the starting
        guess for y1 and y2 as well as the search range.

        new_prev_y should be of the same format as returned by self.prev_y().
        """

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        """Update the previous temperature of the model.

        new_prev_theta should be of the same format as returned by self.prev_theta().
        """

    def update_boundary_traction(self, current_time: float) -> None:
        """Update the forces acting on the boundary."""


def _model(
    grid: Grid,
    initial_temperature: float,
    search_radius: float,
    fps: float,
    boundary_traction: Callable[[float, float], list[float]],
) -> pyo.ConcreteModel:
    """Create model for the mechanical step without regularization."""
    m = pyo.ConcreteModel("Mechanical Step")

    m.vertices = pyo.Set(initialize=grid.vertices, dimen=1)
    m.neumann_vertices = pyo.Set(initialize=grid.neumann_boundary.vertices, dimen=1)
    m.dirichlet_vertices = pyo.Set(initialize=grid.dirichlet_boundary.vertices, dimen=1)

    m.prev_y1 = pyo.Param(
        m.vertices,
        within=pyo.Reals,
        initialize=lambda _, i: grid.points[i][0],
        mutable=True,
    )
    m.prev_y2 = pyo.Param(
        m.vertices,
        within=pyo.Reals,
        initialize=lambda _, i: grid.points[i][1],
        mutable=True,
    )
    m.prev_theta = pyo.Param(
        m.vertices,
        within=pyo.NonNegativeReals,
        initialize=initial_temperature,
        mutable=True,
    )

    m.g1 = pyo.Param(
        m.neumann_vertices,
        within=pyo.Reals,
        initialize=lambda _, i: boundary_traction(*grid.points[i])[0],
        mutable=True,
    )
    m.g2 = pyo.Param(
        m.neumann_vertices,
        within=pyo.Reals,
        initialize=lambda _, i: boundary_traction(*grid.points[i])[1],
        mutable=True,
    )

    m.y1 = pyo.Var(m.vertices, within=pyo.Reals)
    m.y2 = pyo.Var(m.vertices, within=pyo.Reals)
    for v in m.vertices:
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

    for v in m.dirichlet_vertices:
        m.y1[v].fix()
        m.y2[v].fix()

    prev_y1 = P1Interpolation(grid, m.prev_y1)
    prev_y2 = P1Interpolation(grid, m.prev_y2)
    prev_deform = Deformation(prev_y1, prev_y2)

    prev_temp = P1Interpolation(grid, m.prev_theta)

    y1 = P1Interpolation(grid, m.y1)
    y2 = P1Interpolation(grid, m.y2)
    deform = Deformation(y1, y2)

    g1 = P1BoundaryInterpolation(grid.neumann_boundary, m.g1)
    g2 = P1BoundaryInterpolation(grid.neumann_boundary, m.g2)

    def boundary_traction(edge, t):
        return Vector([g1(edge, t), g2(edge, t)])

    integrator = Integrator(DUNAVANT2, grid.triangles, grid.points)
    integrator_for_piecewise_constant = Integrator(CENTROID, grid.triangles, grid.points)
    neumann_boundary_integrator = BoundaryIntegrator(1, grid.neumann_boundary.edges, grid.points)

    m.total_elastic_energy = integrator(
        compose_to_integrand(total_elastic_potential, deform.strain, prev_temp)
    )
    m.dissipation = integrator_for_piecewise_constant(
        compose_to_integrand(dissipation_potential, prev_deform.strain, deform.strain)
    )
    m.boundary_traction_energy = neumann_boundary_integrator(
        compose_to_integrand(boundary_traction_potential, boundary_traction, deform.on_edge)
    )
    m.objective = pyo.Objective(
        expr=m.total_elastic_energy - m.boundary_traction_energy + fps * m.dissipation
    )

    return m


def boundary_traction_potential(boundary_traction, deform):
    return boundary_traction.dot(deform)


class _MechanicalStep(AbstractMechanicalStep):
    """Mechanical step without regularization using P1 finite elements for the deformation."""

    def __init__(
        self,
        solver,
        grid: Grid,
        initial_temperature: float,
        search_radius: float,
        fps: float,
        boundary_traction: Callable[[float, float, float], list[float]],
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._search_radius = search_radius
        self._model = _model(
            grid, initial_temperature, search_radius, fps, lambda x, y: boundary_traction(0, x, y)
        )
        self._boundary_traction = boundary_traction
        self._grid = grid

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        """Return the previous deformation as Nx2 numpy array, where N is the number of vertices."""
        m = self._model
        return np.array([[m.prev_y1[i].value, m.prev_y2[i].value] for i in m.vertices])

    def y(self) -> npt.NDArray[np.float64]:
        """Return the current deformation as Nx2 numpy array, where N is the number of vertices."""
        m = self._model
        return np.array([[m.y1[i].value, m.y2[i].value] for i in m.vertices])

    def prev_theta(self) -> npt.NDArray[np.float64]:
        """Return the previous temperature as vector of length N, where N is the number of vertices."""
        m = self._model
        return np.array([m.prev_theta[i].value for i in m.vertices])

    def update_prev_y(self, new_prev_y: npt.NDArray[np.float64]) -> None:
        if new_prev_y.shape != (self._num_vertices, 2):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_y.shape} (should be"
                f" {(self._num_vertices, 2)})"
            )

        new_prev_y1 = new_prev_y[:, 0]
        new_prev_y2 = new_prev_y[:, 1]
        m = self._model
        for i in m.vertices:
            m.prev_y1[i] = new_prev_y1[i]
            m.prev_y2[i] = new_prev_y2[i]

            m.y1[i] = new_prev_y1[i]
            m.y1[i].bounds = (
                new_prev_y1[i] - self._search_radius,
                new_prev_y1[i] + self._search_radius,
            )
            m.y2[i] = new_prev_y2[i]
            m.y2[i].bounds = (
                new_prev_y2[i] - self._search_radius,
                new_prev_y2[i] + self._search_radius,
            )

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        if new_prev_theta.shape != (self._num_vertices,):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_theta.shape} (should be"
                f" {(self._num_vertices,)})"
            )

        m = self._model
        for i in m.vertices:
            m.prev_theta[i] = new_prev_theta[i]

    def update_boundary_traction(self, current_time: float) -> None:
        m = self._model
        for i in m.neumann_vertices:
            m.g1[i], m.g2[i] = self._boundary_traction(current_time, *self._grid.points[i])


def _model_regularized(
    grid: Grid,
    refined_grid: RefinedGrid,
    initial_temperature: float,
    search_radius: float,
    fps: float,
    regularization: float,
) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel("Mechanical Step with Regularization")
    m.vertices = pyo.RangeSet(len(grid.vertices))
    m.c1_indices = pyo.RangeSet(6)
    m.dirichlet_edges = pyo.RangeSet(len(grid.dirichlet_boundary.edges))
    m.deformation_indices = m.vertices * m.c1_indices
    m.dirichlet_constraint_indices = m.dirichlet_edges * m.c1_indices
    m.refined_vertices = pyo.RangeSet(len(refined_grid.vertices))

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
    m.prev_theta = pyo.Param(
        m.refined_vertices,
        within=pyo.NonNegativeReals,
        initialize=initial_temperature,
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

    prev_y1_params = [[m.prev_y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_y1 = RefinedInterpolation(C1Interpolation(grid, prev_y1_params), refined_grid)
    prev_y2_params = [[m.prev_y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    prev_y2 = RefinedInterpolation(C1Interpolation(grid, prev_y2_params), refined_grid)
    prev_deform = Deformation(prev_y1, prev_y2)

    prev_temp = P1Interpolation(refined_grid, [m.prev_theta[i] for i in list(m.refined_vertices)])

    y1_params = [[m.y1[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y1 = RefinedInterpolation(C1Interpolation(grid, y1_params), refined_grid)
    y2_params = [[m.y2[i, j] for j in list(m.c1_indices)] for i in list(m.vertices)]
    y2 = RefinedInterpolation(C1Interpolation(grid, y2_params), refined_grid)
    deform = Deformation(y1, y2)

    integrator = Integrator(DUNAVANT5, refined_grid.triangles, refined_grid.points)

    m.total_elastic_energy = integrator(
        compose_to_integrand(total_elastic_potential, deform.strain, prev_temp)
    )
    m.hyper_elastic_energy = integrator(
        compose_to_integrand(hyper_elastic_potential, deform.hyper_strain)
    )
    m.dissipation = integrator(
        compose_to_integrand(dissipation_potential, prev_deform.strain, deform.strain)
    )
    m.objective = pyo.Objective(
        expr=m.total_elastic_energy + regularization * m.hyper_elastic_energy + fps * m.dissipation
    )

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
        refined_grid: RefinedGrid,
        initial_temperature: float,
        search_radius: float,
        fps: float,
        regularization: float,
    ):
        self._solver = solver
        self._num_vertices = len(grid.vertices)
        self._search_radius = search_radius
        self._model = _model_regularized(
            grid,
            refined_grid,
            initial_temperature,
            search_radius,
            fps,
            regularization,
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

                m.y1[i, j] = new_prev_y1[i - 1, j - 1]
                m.y1[i, j].bounds = (
                    new_prev_y1[i - 1, j - 1] - self._search_radius,
                    new_prev_y1[i - 1, j - 1] + self._search_radius,
                )
                m.y2[i, j] = new_prev_y2[i - 1, j - 1]
                m.y2[i, j].bounds = (
                    new_prev_y2[i - 1, j - 1] - self._search_radius,
                    new_prev_y2[i - 1, j - 1] + self._search_radius,
                )

    def update_prev_theta(self, new_prev_theta: npt.NDArray[np.float64]) -> None:
        refined_vertices = list(self._model.refined_vertices)
        if new_prev_theta.shape != (len(refined_vertices),):
            raise ValueError(
                f"Input array has incorrect shape {new_prev_theta.shape} (should be"
                f" {(len(refined_vertices),)})"
            )

        for i in refined_vertices:
            self._model.prev_theta[i] = new_prev_theta[i - 1]


def mechanical_step(
    solver,
    grid: Grid,
    params: MechanicalStepParams,
    refined_grid: RefinedGrid | None,
    boundary_traction: Callable[[float, float, float], list[float]],
) -> AbstractMechanicalStep:
    """Mechanical step factory."""
    if params.regularization is None:
        return _MechanicalStep(
            solver,
            grid,
            params.initial_temperature,
            params.search_radius,
            params.fps,
            boundary_traction,
        )

    if refined_grid is None:
        raise ValueError("A refined grid must be provided for a regularized model")

    return _MechanicalStepRegularized(
        solver,
        grid,
        refined_grid,
        params.initial_temperature,
        params.search_radius,
        params.fps,
        params.regularization,
    )
