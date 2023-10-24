from tvemoves_rufbad.interpolation import C1Interpolation
from tvemoves_rufbad.tensors import Vector, Matrix
from tvemoves_rufbad.grid import SquareEquilateralGrid


def f(x: float, y: float) -> float:
    return (x - 1 / 2) ** 2 + (y - 1 / 2) ** 2


def grad_f(x: float, y: float) -> Vector:
    return Vector([2 * (x - 1 / 2), 2 * (y - 1 / 2)])


def hessian_f(x: float, y: float) -> Matrix:
    return Matrix([[2, 0], [0, 2]])


grid = SquareEquilateralGrid(num_horizontal_points=4)
f_at_grid_points = [f(p[0], p[1]) for p in grid.points]
grad_f_at_grid_points = [grad_f(p[0], p[1]) for p in grid.points]
hessian_f_at_grid_points = [hessian_f(p[0], p[1]) for p in grid.points]
params = [
    [f, G[0], G[1], H[0, 0], H[0, 1], H[1, 1]]
    for (f, G, H) in zip(
        f_at_grid_points, grad_f_at_grid_points, hessian_f_at_grid_points
    )
]
f_approx = C1Interpolation(grid, params)
print(f_approx.on_edge(grid.edges[0], 1))
