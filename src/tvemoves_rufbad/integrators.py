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
            abs(first.vstack(second).det() / 2)
            for (first, second) in zip(first_sides, second_sides)
        ]

    def __call__(self, integrand):
        # integrand(triangle, (b1, b2)) should return a float,
        # where triangle is any triangle in self._triangles
        # and (b1, b2, 1 - b1 - b2) are barycentric coordinates
        return sum(
            triangle_area
            * sum(
                weight * integrand(triangle, point)
                for (point, weight) in zip(
                    self._quadrature.points, self._quadrature.weights
                )
            )
            for (triangle_area, triangle) in zip(self._triangle_areas, self._triangles)
        )
