import pyomo.environ as pyo


class MechanicalStep:
    def __init__(self, grid, initial_temperature, search_radius):
        self._grid = grid
        self._model = pyo.ConcreteModel()
        m = self._model

        m.prev_x1 = pyo.Param(
            grid.vertices,
            within=pyo.Reals,
            initialize=[p[0] for p in grid.initial_positions],
            mutable=True,
        )
        m.prev_x2 = pyo.Param(
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

        m.x1 = pyo.Var(grid.vertices, within=pyo.Reals)
        m.x2 = pyo.Var(grid.vertices, within=pyo.Reals)
        for v in grid.vertices:
            m.x1[v] = m.prev_x1[v]
            m.x1[v].bounds = (
                m.prev_x1[v] - search_radius,
                m.prev_x1[v] + search_radius,
            )
            m.x2[v] = m.prev_x2[v]
            m.x2[v].bounds = (
                m.prev_x2[v] - search_radius,
                m.prev_x2[v] + search_radius,
            )

        for v in grid.dirichlet_vertices:
            m.x1[v].fix()
            m.x2[v].fix()
