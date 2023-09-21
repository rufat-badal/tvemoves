import pyomo.environ as pyo
from .interpolation import P1Deformation, P1Interpolation


class MechanicalStep:
    def __init__(self, grid, initial_temperature, search_radius):
        self._grid = grid
        self._model = pyo.ConcreteModel()
        m = self._model

        m.prev_y1 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[0] for p in grid.initial_positions],
            mutable=True,
        )
        m.prev_y2 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[1] for p in grid.initial_positions],
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

        prev_deform = P1Deformation(grid, m.prev_y1, m.prev_y2)
        deform = P1Deformation(grid, m.y1, m.y2)
        prev_temp = P1Interpolation(grid, m.prev_theta)
