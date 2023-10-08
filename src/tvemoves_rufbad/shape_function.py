from .tensors import Vector, Matrix
import sympy as sp

L1, L2, L3 = sp.symbols("L1 L2 L3")
L = [L1, L2, L3]
t = sp.symbols("t")
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
shape_function_segment_symbolic = [
    Ni.subs(L1, t).subs(L2, 1 - t).subs(L3, 0) for Ni in shape_function_symbolic
]

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


shape_function_segment_lambdified = sp.lambdify(
    [t, b3, c3], shape_function_segment_symbolic
)


def shape_function_segment(t: float, b3: float, c3: float) -> Vector:
    return Vector(shape_function_segment_lambdified(t, b3, c3))


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
shape_function_hessian_vectorized_symbolic = sp.Matrix(
    [
        [H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 2]]
        for H in shape_function_hessian_symbolic
    ]
)
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
) -> Matrix:
    return Matrix(
        shape_function_hessian_vectorized_lambdified(
            L1, L2, L3, b1, b2, b3, c1, c2, c3
        ).tolist()
    )
