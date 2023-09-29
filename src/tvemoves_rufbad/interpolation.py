from .tensors import Vector, Matrix
from dataclasses import dataclass
from .grid import Grid


class P1Interpolation:
    def __init__(self, grid: Grid, params: list[float]):
        if len(params) != len(grid.vertices):
            raise ValueError("number of params must equal to the number of vertices")
        self._grid = grid
        self._params = params

    def __call__(
        self,
        triangle: tuple[int, int, int],
        barycentric_coords: tuple[float, float, float],
    ):
        i1, i2, i3 = triangle
        t1, t2, t3 = barycentric_coords
        return t1 * self._params[i1] + t2 * self._params[i2] + t3 * self._params[i3]

    def boundary(self, segment: tuple[int, int], t: float):
        i1, i2 = segment
        return t * self._params[i1] + (1 - t) * self._params[i2]

    def gradient(self, triangle: tuple[int, int, int], barycentric_coordinates=None):
        i1, i2, i3 = triangle
        barycentric_gradient = Vector(
            [self._params[i1], self._params[i2], self._params[i3]]
        )
        return self._grid.gradient_transform(triangle, barycentric_gradient)


class P1Deformation:
    def __init__(self, grid, y1_params: list[float], y2_params: list[float]):
        self.y1 = P1Interpolation(grid, y1_params)
        self.y2 = P1Interpolation(grid, y2_params)

    def __call__(
        self,
        triangle: tuple[int, int, int],
        barycentric_coords: tuple[float, float, float],
    ):
        return Vector(
            [
                self.y1(triangle, barycentric_coords),
                self.y2(triangle, barycentric_coords),
            ]
        )

    def strain(self, triangle: tuple[int, int, int], barycentric_coordinates=None):
        return self.y1.gradient(triangle).vstack(self.y2.gradient(triangle))


class C1Interpolation:
    def __init__(self, grid: Grid, params: list[list[float]]):
        self._grid = grid
        # each param is a list of 6 floats corresponding to
        # u, u_x, u_y, u_xx, u_xy, u_yy, where u denotes the function
        # wish to interpolate
        self._params = [Vector(p) for p in params]

    def __call__(
        self,
        triangle: tuple[int, int, int],
        barycentric_coords: tuple[float, float, float],
    ):
        i1, i2, i3 = triangle
        L1, L2, L3 = barycentric_coords
        Ns = [
            self._grid.shape_function((i1, i2, i3), (L1, L2, L3)),
            self._grid.shape_function((i2, i3, i1), (L2, L3, L1)),
            self._grid.shape_function((i3, i1, i2), (L3, L1, L2)),
        ]
        ps = [self._params[i1], self._params[i2], self._params[i3]]
        return sum(N.dot(p) for (N, p) in zip(Ns, ps))
