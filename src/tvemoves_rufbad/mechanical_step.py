import pyomo.environ as pyo
import numpy as np
from .interpolation import P1Deformation, P1Interpolation
from .quadrature_rules import DUNAVANT2, CENTROID
from .tensors import Matrix
from .integrators import Integrator
from .grid import Grid
from .utils import (
    generate_martensite_potential,
    austenite_percentage,
    austenite_potential,
    compose_to_integrand,
    dissipation_norm,
    symmetrized_strain_delta,
)
import numpy.typing as npt
import numpy as np


class MechanicalStep:
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
        self._model = pyo.ConcreteModel("Mechanical Step")
        m = self._model
        self._num_vertices = len(grid.vertices)

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

        for v in grid.dirichlet_vertices:
            m.y1[v].fix()
            m.y2[v].fix()

        integrator = Integrator(DUNAVANT2, grid.triangles, grid.points)
        integrator_for_piecewise_constant = Integrator(
            CENTROID, grid.triangles, grid.points
        )

        prev_deform = P1Deformation(grid, m.prev_y1, m.prev_y2)
        deform = P1Deformation(grid, m.y1, m.y2)
        prev_temp = P1Interpolation(grid, m.prev_theta)

        # total elastic energy
        scaling_matrix = Matrix([[1 / shape_memory_scaling, 0], [0, 1]])
        martensite_potential = generate_martensite_potential(scaling_matrix)

        martensite_percentage = lambda theta: 1 - austenite_percentage(theta)
        total_elastic_potential = lambda F, theta: (
            austenite_percentage(theta) * austenite_potential(F)
            + martensite_percentage(theta) * martensite_potential(F)
        )
        total_elastic_integrand = compose_to_integrand(
            total_elastic_potential, deform.strain, prev_temp
        )
        m.total_elastic_energy = integrator(total_elastic_integrand)

        # dissipation
        dissipation_potential = lambda prev_F, F: dissipation_norm(
            symmetrized_strain_delta(prev_F, F)
        )
        dissipation_integrand = compose_to_integrand(
            dissipation_potential, prev_deform.strain, deform.strain
        )
        m.dissipation = integrator_for_piecewise_constant(dissipation_integrand)

        m.objective = pyo.Objective(expr=m.total_elastic_energy + fps * m.dissipation)

    def solve(self) -> None:
        self._solver.solve(self._model)

    def prev_y(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [self._model.prev_y1[i].value, self._model.prev_y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def y(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [self._model.y1[i].value, self._model.y2[i].value]
                for i in range(self._num_vertices)
            ]
        )

    def prev_theta(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self._model.prev_theta[i].value for i in range(self._num_vertices)]
        )
