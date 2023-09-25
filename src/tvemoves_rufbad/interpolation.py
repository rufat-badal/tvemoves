from .tensors import Vector


class P1Interpolation:
    def __init__(self, grid, params):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(self, triangle, barycentric_coords):
        i1, i2, i3 = triangle
        t1, t2 = barycentric_coords
        t3 = 1 - t1 - t2
        return t1 * self._params[i1] + t2 * self._params[i2] + t3 * self._params[i3]

    def boundary(self, segment, t1):
        i1, i2 = segment
        t2 = 1 - t1
        return t1 * self._params[i1] + t2 * self._params[i2]

    def gradient(self, triangle):
        i1, i2, i3 = triangle
        barycentric_gradient = Vector(
            [self._params[i1], self._params[i2], self._params[i3]]
        )
        return self._grid.gradient_transform(triangle, barycentric_gradient)


class P1Deformation:
    def __init__(self, grid, y1_params, y2_params):
        self.y1 = P1Interpolation(grid, y1_params)
        self.y2 = P1Interpolation(grid, y2_params)

    def __call__(self, triangle, barycentric_coords):
        return Vector(
            [
                self.y1(triangle, barycentric_coords),
                self.y2(triangle, barycentric_coords),
            ]
        )

    def strain(self, triangle, barycentric_coordinates=None):
        return self.y1.gradient(triangle).vstack(self.y2.gradient(triangle))
