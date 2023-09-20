from .tensors import Vector


class P1Interpolator:
    def __init__(self, grid, params):
        # params: single scalar value for each grid point
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(self, triangle, barycentric_coords):
        i1, i2, i3 = triangle
        t1, t2 = barycentric_coords
        t3 = 1 - t2 - t3
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
        return self.grid.gradient_transform(triangle, barycentric_gradient)


class P1Deformation:
    def __init__(self, x1_params, x2_params):
        self.y1 = P1Interpolator(x1_params)
        self.y2 = P1Interpolator(x2_params)

    def __call__(self, triangle, barycentric_coords):
        return Vector(
            [
                self.y1(triangle, barycentric_coords),
                self.y2(triangle, barycentric_coords),
            ]
        )
