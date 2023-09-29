from .tensors import Vector, Matrix
import sympy as sp
from dataclasses import dataclass

# shape function and its derivatives
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
shape_function_lambdified = sp.lambdify(L + b + c, shape_function_symbolic)


def shape_function(
    L1: float,
    L2: float,
    L3: float,
    b1: float,
    b2: float,
    b3: float,
    c1: float,
    c2: float,
    c3: float,
) -> Vector:
    return Vector(shape_function_lambdified(L1, L2, L3, b1, b2, b3, c1, c2, c3))


shape_function_jacobian_symbolic = sp.Matrix(
    [[sp.diff(shape_function_symbolic[i], L[j]) for j in range(3)] for i in range(6)]
)
shape_function_jacobian_lambdified = sp.lambdify(
    L + b + c, shape_function_jacobian_symbolic
)


def shape_function_jacobian(
    L1: float,
    L2: float,
    L3: float,
    b1: float,
    b2: float,
    b3: float,
    c1: float,
    c2: float,
    c3: float,
) -> Matrix:
    return Matrix(
        shape_function_jacobian_lambdified(L1, L2, L3, b1, b2, b3, c1, c2, c3).tolist()
    )


shape_function_hessian_symbolic = sp.Array(
    [
        [
            [sp.diff(shape_function_symbolic[i], L[j], L[k]) for k in range(3)]
            for j in range(3)
        ]
        for i in range(6)
    ]
)
shape_function_hessian_vectorized_symbolic = [
    H[idx]
    for H in shape_function_hessian_symbolic
    for idx in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
]
shape_function_hessian_vectorized_lambdified = sp.lambdify(
    L + b + c, shape_function_hessian_vectorized_symbolic
)


def shape_function_hessian_vectorized(
    L1: float,
    L2: float,
    L3: float,
    b1: float,
    b2: float,
    b3: float,
    c1: float,
    c2: float,
    c3: float,
) -> Vector:
    return Vector(
        shape_function_hessian_vectorized_lambdified(L1, L2, L3, b1, b2, b3, c1, c2, c3)
    )


@dataclass(frozen=True)
class Grid:
    vertices: list[int]
    edges: list[tuple[int, int]]
    triangles: list[tuple[int, int, int]]
    boundary_vertices: list[int]
    boundary_edges: list[tuple[int, int]]
    dirichlet_vertices: list[int]
    dirichlet_edges: list[tuple[int, int]]
    neumann_vertices: list[int]
    neumann_edges: list[tuple[int, int]]
    points: list[Vector]

    def _triangle_parameters(
        self, triangle: tuple[int, int, int]
    ) -> tuple[list[float], list[float], float]:
        i, j, k = triangle
        x, y, z = (
            self.points[i],
            self.points[j],
            self.points[k],
        )
        a = [
            y[0] * z[1] - z[0] * y[1],
            z[0] * x[1] - x[0] * z[1],
            x[0] * y[1] - y[0] * x[1],
        ]
        b = [y[1] - z[1], z[1] - x[1], x[1] - y[1]]
        c = [z[0] - y[0], x[0] - z[0], y[0] - x[0]]
        delta = (a[0] + a[1] + a[2]) / 2

        return b, c, delta

    def gradient_transform(
        self,
        triangle: tuple[int, int, int],
        barycentric_gradient: Vector,
    ) -> Vector:
        b, c, delta = self._triangle_parameters(triangle)
        trafo_matrix = Matrix([[b[0], b[1], b[2]], [c[0], c[1], c[2]]]) / (2 * delta)
        return trafo_matrix.dot(barycentric_gradient)

    def hessian_transform(
        self,
        triangle: tuple[int, int, int],
        barycentric_hessian: Matrix,
    ) -> Matrix:
        b, c, delta = self._triangle_parameters(triangle)
        trafo_matrix = Matrix(
            [
                [
                    b[0] ** 2,
                    b[1] ** 2,
                    b[2] ** 2,
                    2 * b[0] * b[1],
                    2 * b[0] * b[2],
                    2 * b[1] * b[2],
                ],
                [
                    c[0] ** 2,
                    c[1] ** 2,
                    c[2] ** 2,
                    2 * c[0] * c[1],
                    2 * c[0] * c[2],
                    2 * c[1] * c[2],
                ],
                [
                    b[0] * c[0],
                    b[1] * c[1],
                    b[2] * c[2],
                    b[0] * c[1] + b[1] * c[0],
                    b[0] * c[2] + b[2] * c[0],
                    b[1] * c[2] + b[2] * c[1],
                ],
            ]
        ) / (4 * delta**2)
        flat_hessian = trafo_matrix.dot(barycentric_hessian.flatten())
        return Matrix(
            [[flat_hessian[0], flat_hessian[2]], [flat_hessian[2], flat_hessian[1]]]
        )

    def shape_function(
        self,
        triangle: tuple[int, int, int],
        barycentric_coordinates: tuple[float, float, float],
    ) -> Vector:
        b, c, _ = self._triangle_parameters(triangle)
        return shape_function(*barycentric_coordinates, *b, *c)

    def shape_function_jacobian(
        self,
        triangle: tuple[int, int, int],
        barycentric_coordinates: tuple[float, float, float],
    ) -> Matrix:
        b, c, _ = self._triangle_parameters(triangle)
        return shape_function_jacobian(*barycentric_coordinates, *b, *c)

    def shape_function_hessian_vectorized(
        self,
        triangle: tuple[int, int, int],
        barycentric_coordinates: tuple[float, float, float],
    ) -> Vector:
        b, c, _ = self._triangle_parameters(triangle)
        return shape_function_hessian_vectorized(*barycentric_coordinates, *b, *c)


def generate_square_equilateral_grid(
    num_horizontal_points: int, fix: str = "none"
) -> Grid:
    def pair_to_vertex(i, j):
        # (0, 0) -> 0, (1, 0) -> 1, ...,
        # (num_horizontal_points - 1, 0) -> num_horizontal_points -1,
        # (0, 1) -> num_horizontal_points, ...
        return j * num_horizontal_points + i

    vertices = list(range(num_horizontal_points * num_horizontal_points))

    horizontal_edges = [
        (pair_to_vertex(i, j), pair_to_vertex(i + 1, j))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points)
    ]
    vertical_edges = [
        (pair_to_vertex(i, j), pair_to_vertex(i, j + 1))
        for i in range(num_horizontal_points)
        for j in range(num_horizontal_points - 1)
    ]
    diagonal_edges = [
        (pair_to_vertex(i, j), pair_to_vertex(i + 1, j + 1))
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    edges = horizontal_edges + vertical_edges + diagonal_edges

    lower_triangles = [
        (
            pair_to_vertex(i + 1, j),
            pair_to_vertex(i, j),
            pair_to_vertex(i + 1, j + 1),
        )
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    upper_triangles = [
        (
            pair_to_vertex(i, j + 1),
            pair_to_vertex(i + 1, j + 1),
            pair_to_vertex(i, j),
        )
        for i in range(num_horizontal_points - 1)
        for j in range(num_horizontal_points - 1)
    ]
    triangles = lower_triangles + upper_triangles

    lower_boundary_vertices = set(
        pair_to_vertex(i, 0) for i in range(num_horizontal_points)
    )
    lower_boundary_edges = [
        (pair_to_vertex(i, 0), pair_to_vertex(i + 1, 0))
        for i in range(num_horizontal_points - 1)
    ]
    right_boundary_vertices = set(
        pair_to_vertex(num_horizontal_points - 1, i)
        for i in range(num_horizontal_points)
    )
    right_boundary_edges = [
        (
            pair_to_vertex(num_horizontal_points - 1, i),
            pair_to_vertex(num_horizontal_points - 1, i + 1),
        )
        for i in range(num_horizontal_points - 1)
    ]
    upper_boundary_vertices = set(
        pair_to_vertex(i, num_horizontal_points - 1)
        for i in range(num_horizontal_points)
    )
    upper_boundary_edges = [
        (
            pair_to_vertex(i, num_horizontal_points - 1),
            pair_to_vertex(i + 1, num_horizontal_points - 1),
        )
        for i in range(num_horizontal_points - 1)
    ]
    left_boundary_vertices = set(
        pair_to_vertex(0, i) for i in range(num_horizontal_points)
    )
    left_boundary_edges = [
        (pair_to_vertex(0, i), pair_to_vertex(0, i + 1))
        for i in range(num_horizontal_points - 1)
    ]

    boundary_vertices = list(
        lower_boundary_vertices.union(right_boundary_vertices)
        .union(upper_boundary_vertices)
        .union(left_boundary_vertices)
    )
    boundary_edges = (
        lower_boundary_edges
        + right_boundary_edges
        + upper_boundary_edges
        + left_boundary_edges
    )

    match fix:
        case "none":
            dirichlet_vertices = []
            dirichlet_edges = []
            neumann_vertices = boundary_vertices
            neumann_edges = boundary_edges
        case "all":
            dirichlet_vertices = boundary_vertices
            dirichlet_edges = boundary_edges
            neumann_vertices = []
            neumann_edges = []
        case "lower":
            dirichlet_vertices = list(lower_boundary_vertices)
            dirichlet_edges = lower_boundary_edges
            neumann_vertices = list(
                right_boundary_vertices.union(upper_boundary_vertices).union(
                    left_boundary_vertices
                )
            )
            neumann_edges = (
                right_boundary_edges + upper_boundary_edges + left_boundary_edges
            )
        case "right":
            dirichlet_vertices = list(right_boundary_vertices)
            dirichlet_edges = right_boundary_edges
            neumann_vertices = list(
                upper_boundary_vertices.union(left_boundary_vertices).union(
                    lower_boundary_vertices
                )
            )
            neumann_edges = (
                upper_boundary_edges + left_boundary_edges + lower_boundary_edges
            )
        case "upper":
            dirichlet_vertices = list(upper_boundary_vertices)
            dirichlet_edges = upper_boundary_edges
            neumann_vertices = list(
                left_boundary_vertices.union(lower_boundary_vertices).union(
                    right_boundary_vertices
                )
            )
            neumann_edges = (
                left_boundary_edges + lower_boundary_edges + right_boundary_edges
            )
        case "left":
            dirichlet_vertices = list(left_boundary_vertices)
            dirichlet_edges = left_boundary_edges
            neumann_vertices = list(
                lower_boundary_vertices.union(right_boundary_vertices).union(
                    upper_boundary_vertices
                )
            )
            neumann_edges = (
                lower_boundary_edges + right_boundary_edges + upper_boundary_edges
            )
        case _:
            raise ValueError("invalid dirichlet condition provided")

    grid_spacing = 1 / (num_horizontal_points - 1)
    points = [
        Vector(
            [
                v % num_horizontal_points * grid_spacing,
                v // num_horizontal_points * grid_spacing,
            ]
        )
        for v in vertices
    ]

    return Grid(
        vertices,
        edges,
        triangles,
        boundary_vertices,
        boundary_edges,
        dirichlet_vertices,
        dirichlet_edges,
        neumann_vertices,
        neumann_edges,
        points,
    )
