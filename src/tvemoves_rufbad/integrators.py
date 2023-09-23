class Integrator:
    def __init__(self, quadrature, grid):
        self._triangles = grid.triangles
        self._quadrature = quadrature
        triangle_points = grid.initial_positions
        first_sides = (
            triangle_points[j] - triangle_points[i] for (i, j, _) in self._triangles
        )
        second_sides = (
            triangle_points[k] - triangle_points[i] for (i, _, k) in self._triangles
        )
        self._triangle_areas = [
            first.vstack(second).det() / 2
            for (first, second) in zip(first_sides, second_sides)
        ]
