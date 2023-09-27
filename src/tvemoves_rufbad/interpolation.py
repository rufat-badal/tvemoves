from .tensors import Vector, Matrix
import sympy as sp


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


L1, L2, L3 = sp.symbols("L1 L2 L3")
L = [L1, L2, L3]
b1, b2, b3 = sp.symbols("b1 b2 b3")
b = [b1, b2, b3]
c1, c2, c3 = sp.symbols("c1 c2 c3")
c = [c1, c2, c3]
r = sp.Matrix(
    [
        [-(b[i] * b[j] + c[i] * c[j]) / (b[i] ** 2 + c[i] ** 2) for j in range(3)]
        for i in range(3)
    ]
)
N1 = (
    L1**5
    + 5 * L1**4 * L2
    + 5 * L1**4 * L3
    + 10 * L1**3 * L2**2
    + 10 * L1**3 * L3**2
    + 20 * L1**3 * L2 * L3
    + 30 * r[1, 0] * L1**2 * L2 * L3**2
    + 30 * r[2, 0] * L1**2 * L3 * L2**2
)
N2 = (
    c[2] * L1**4 * L2
    - c[1] * L1**4 * L3
    + 4 * c[2] * L1**3 * L2**2
    - 4 * c[1] * L1**3 * L3**2
    + 4 * (c[2] - c[1]) * L1**3 * L2 * L3
    - (3 * c[0] + 15 * r[1, 0] * c[1]) * L1**2 * L2 * L3**2
    + (3 * c[0] + 15 * r[2, 0] * c[2]) * L1**2 * L3 * L2**2
)
N3 = (
    -b[2] * L1**4 * L2
    + b[1] * L1**4 * L3
    - 4 * b[2] * L1**3 * L2**2
    + 4 * b[1] * L1**3 * L3**2
    + 4 * (b[1] - b[2]) * L1**3 * L2 * L3
    + (3 * b[0] + 15 * r[1, 0] * b[1]) * L1**2 * L2 * L3**2
    - (3 * b[0] + 15 * r[2, 0] * b[2]) * L1**2 * L3 * L2**2
)
N4 = (
    c[2] ** 2 / 2 * L1**3 * L2**2
    + c[1] ** 2 / 2 * L1**3 * L3**2
    - c[1] * c[2] * L1**3 * L2 * L3
    + (c[0] * c[1] + 5 / 2 * r[1, 0] * c[1] ** 2) * L2 * L3**2 * L1**2
    + (c[0] * c[2] + 5 / 2 * r[2, 0] * c[2] ** 2) * L3 * L2**2 * L1**2
)
N5 = (
    -b[2] * c[2] * L1**3 * L2**2
    - b[1] * c[1] * L1**3 * L3**2
    + (b[1] * c[2] + b[2] * c[1]) * L1**3 * L2 * L3
    - (b[0] * c[1] + b[1] * c[0] + 5 * r[1, 0] * b[1] * c[1]) * L2 * L3**2 * L1**2
    - (b[0] * c[2] + b[2] * c[0] + 5 * r[2, 0] * b[2] * c[2]) * L3 * L2**2 * L1**2
)
N6 = (
    b[2] ** 2 / 2 * L1**3 * L2**2
    + b[1] ** 2 / 2 * L1**3 * L3**2
    - b[1] * b[2] * L1**3 * L2 * L3
    + (b[0] * b[1] + 5 / 2 * r[1, 0] * b[1] ** 2) * L2 * L3**2 * L1**2
    + (b[0] * b[2] + 5 / 2 * r[2, 0] * b[2] ** 2) * L3 * L2**2 * L1**2
)

shape_function_symbolic = [N1, N2, N3, N4, N5, N6]
shape_function = sp.lambdify(L + b + c, shape_function_symbolic)

jacobian_of_shape_function_symbolic = sp.Matrix(
    [[sp.diff(shape_function_symbolic[i], L[j]) for j in range(3)] for i in range(6)]
)
jacobian_of_shape_function_lambdified = sp.lambdify(
    L + b + c, jacobian_of_shape_function_symbolic
)


def jacobian_of_shape_function(L1, L2, L3, b1, b2, b3, c1, c2, c3):
    return Matrix(
        jacobian_of_shape_function_lambdified(
            L1, L2, L3, b1, b2, b3, c1, c2, c3
        ).tolist()
    )
