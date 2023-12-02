"""Implementation of 1-d, 2-d, and 3-d tensors."""

from __future__ import annotations
from itertools import permutations
from typing import Callable, Iterator
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo


class Vector:
    """Custom vector class."""

    def __init__(self, data: list) -> None:
        self.size = len(data)
        self.shape = (self.size,)
        self.data = data

    def __repr__(self) -> str:
        return (
            "Vector([" + ", ".join([str(self.data[i]) for i in range(self.size)]) + "])"
        )

    def __str__(self) -> str:
        return "[" + " ".join([str(self.data[i]) for i in range(self.size)]) + "]"

    def __getitem__(self, i: int | slice):
        if isinstance(i, int):
            if i < 0 or i >= self.size:
                raise IndexError(
                    f"index {i} is out of bounds for vector of size {self.size}"
                )
            return self.data[i]
        else:
            return Vector(self.data[i])

    def __neg__(self) -> Vector:
        return Vector([-x for x in self.data])

    def __add__(self, other: Vector) -> Vector:
        if self.size != other.size:
            raise ValueError(
                f"vectors of size {self.size} and {other.size} cannot be added"
            )
        return Vector([x + y for x, y in zip(self.data, other.data)])

    def __sub__(self, other: Vector) -> Vector:
        if self.size != other.size:
            raise ValueError(
                f"vectors of sizes {self.size} and {other.size} cannot be subtracted"
            )
        return Vector([x - y for x, y in zip(self.data, other.data)])

    def __mul__(self, scaling) -> Vector:
        return Vector([x * scaling for x in self.data])

    def __rmul__(self, scaling) -> Vector:
        return self.__mul__(scaling)

    def __truediv__(self, divisor) -> Vector:
        return Vector([x / divisor for x in self.data])

    def __rtruediv__(self, divisor):
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.data == other.data

    def normsqr(self):
        """Compute square of the Euclidean norm."""
        return sum(x * x for x in self.data)

    def norm(self):
        """Compute square of the Euclidean norm."""
        return pyo.sqrt(self.normsqr())

    def dot(self, other: Vector):
        """Compute the scalar product with another vector."""
        return sum(x * y for x, y in zip(self.data, other.data))

    def map(self, f: Callable) -> Vector:
        """Apply a map f to each entry in the vector."""
        return Vector([f(x) for x in self.data])

    def stack(self, other: Vector) -> Matrix:
        """Vertically stack two vectors to create a matrix with two rows."""
        if self.size != other.size:
            raise ValueError(
                f"cannot stack vectors of sizes {self.size} and {other.size}"
            )
        return Matrix(
            [
                self.data,
                other.data,
            ]
        )

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def reshape(self, num_rows: int, num_cols: int) -> Matrix:
        """Reshape vector of length num_rows x num_cols to a num_rows x num_cols matrix."""
        if self.size != num_rows * num_cols:
            raise ValueError(
                f"cannot reshape vector of size {self.size} to a matrix of shape ({num_rows}, {num_cols})"
            )
        return Matrix(
            [
                [self.data[i * num_cols + j] for j in range(num_cols)]
                for i in range(num_rows)
            ]
        )

    def numpy(self) -> npt.NDArray:
        """Return copy of the vector as numpy array."""
        return np.array(self.data)

    def extend(self, other: Vector) -> Vector:
        """Extend vector with components of another vector."""
        return Vector(self.data + other.data)

    def __len__(self) -> int:
        return self.size


def sign(p):
    """Compute the sign of a permutation."""
    num_misplaced = 0
    for i, a in enumerate(p):
        for b in p[i + 1 :]:
            if b < a:
                num_misplaced += 1

    return 1 if num_misplaced % 2 == 0 else -1


class Matrix:
    """Custom matrix class."""

    def __init__(self, data: list[list]):
        self.shape = (len(data), len(data[0]))
        for row in data[1:]:
            if len(row) != self.shape[1]:
                raise ValueError("inhomogeneously shaped initialization list provided")
        self.data = data

    def __repr__(self):
        lines = [
            "[" + ", ".join([repr(self.data[i][j]) for j in range(self.shape[1])]) + "]"
            for i in range(self.shape[0])
        ]
        typeinfo = "Matrix(["
        data = (",\n" + len(typeinfo) * " ").join(lines)
        return typeinfo + data + "])"

    def __str__(self):
        lines = [
            "[" + " ".join([repr(self.data[i][j]) for j in range(self.shape[1])]) + "]"
            for i in range(self.shape[0])
        ]
        start = "["
        data = ("\n" + len(start) * " ").join(lines)
        return start + data + "]"

    def __getitem__(self, index: tuple[int, int]):
        i, j = index
        if i < 0 or i >= self.shape[0]:
            raise IndexError(
                f"row index {i} out of bounds for matrix with {self.shape[0]} rows"
            )
        if j < 0 or j >= self.shape[1]:
            raise IndexError(
                f"column index {j} out of bounds for matrix with {self.shape[1]} columns"
            )
        return self.data[i][j]

    def __neg__(self) -> Matrix:
        return Matrix([[-x for x in row] for row in self.data])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.data == other.data

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(
                f"matrices of shapes {self.shape} and {other.shape} cannot be added"
            )
        return Matrix(
            [
                [x + y for x, y in zip(row, other_row)]
                for row, other_row in zip(self.data, other.data)
            ]
        )

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(
                f"matrices of shapes {self.shape} and {other.shape} cannot be subtracted"
            )
        return Matrix(
            [
                [x - y for x, y in zip(row, other_row)]
                for row, other_row in zip(self.data, other.data)
            ]
        )

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"matrices of shapes {self.shape} and {other.shape} cannot be multiplied"
            )
        return Matrix(
            [
                [
                    sum(
                        self.data[i][k] * other.data[k][j] for k in range(self.shape[1])
                    )
                    for j in range(other.shape[1])
                ]
                for i in range(self.shape[0])
            ]
        )

    def transpose(self) -> Matrix:
        """Compute the transpose of a matrix."""
        return Matrix(
            [
                [self.data[i][j] for i in range(self.shape[0])]
                for j in range(self.shape[1])
            ]
        )

    def trace(self):
        """Compute the trace of a matrix."""
        return sum(self.data[i][i] for i in range(min(self.shape)))

    def det(self):
        """Compute the determinant of a matrix via Laplace expansion."""
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                f"det not defined for nonsquare matrix of shape {self.shape}"
            )
        res = 0
        for p in permutations(range(self.shape[0])):
            prod = 1
            for i in range(self.shape[0]):
                prod *= self.data[i][p[i]]
            res += sign(p) * prod

        return res

    def normsqr(self):
        """Compute the square of the Frobenius norm of a matrix."""
        return sum(x**2 for row in self.data for x in row)

    def norm(self):
        """Compute the the Frobenius norm of a matrix."""
        return pyo.sqrt(self.normsqr())

    def scalar_product(self, other):
        """Compute the Frobenius scalar product with another matrix."""
        if self.shape != other.shape:
            raise ValueError(
                f"matrices of shape {self.shape} and {other.shape} cannot be scalar multiplied"
            )
        return sum(
            x * y
            for row, other_row in zip(self.data, other.data)
            for (x, y) in zip(row, other_row)
        )

    def __mul__(self, scaling) -> Matrix:
        """Compute scaled matrix."""
        return Matrix([[x * scaling for x in row] for row in self.data])

    def __rmul__(self, scaling) -> Matrix:
        return self.__mul__(scaling)

    def __truediv__(self, divisor) -> Matrix:
        return Matrix([[x / divisor for x in row] for row in self.data])

    def __rtruediv__(self, divisor):
        return NotImplemented

    def flatten(self) -> Vector:
        """Returns matrix flattened to a vector."""
        return Vector([x for row in self.data for x in row])

    def dot(self, v) -> Vector:
        """Compute matrix-vector-product."""
        if v.size != self.shape[1]:
            raise ValueError(
                f"cannot multiply a matrix of shape {self.shape} with a vector of size {v.size}"
            )
        return Vector(
            [
                sum(self.data[i][j] * v[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ]
        )

    def map(self, f: Callable) -> Matrix:
        """Applies map f to each entry of the matrix."""
        return Matrix([[f(x) for x in row] for row in self.data])

    def stack(self, other: Matrix) -> Tensor3D:
        """Stack two matrices to a 3-d tensor."""
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                f"matrices of shapes {self.shape} and {other.shape} cannot be stacked"
            )
        return Tensor3D(
            [
                self.data,
                other.data,
            ]
        )

    def numpy(self) -> npt.NDArray:
        """Return copy of the matrix as numpy array."""
        return np.array(self.data)


class Tensor3D:
    """Custom 3-d tensor class."""

    def __init__(self, data: list[list[list]]):
        self.shape = (len(data), len(data[0]), len(data[0][0]))
        for submatrix in data:
            if len(submatrix) != self.shape[1]:
                raise ValueError("inhomogeneously shaped initialization list provided")
            for row in submatrix:
                if len(row) != self.shape[2]:
                    raise ValueError(
                        "inhomogeneously shaped initialization list provided"
                    )
        self.data = data

    def __repr__(self):
        typeinfo = "Tensor3D(["
        matrices = []
        for i in range(self.shape[0]):
            matrix_rows = [
                "["
                + ", ".join([repr(self.data[i][j][k]) for k in range(self.shape[2])])
                + "]"
                for j in range(self.shape[1])
            ]
            matrices.append(
                "[" + (",\n " + len(typeinfo) * " ").join(matrix_rows) + "]"
            )
        data = (",\n" + len(typeinfo) * " ").join(matrices)
        return typeinfo + data + "])"

    def __str__(self):
        start = "["
        matrices = []
        for i in range(self.shape[0]):
            matrix_rows = [
                "["
                + ", ".join([repr(self.data[i][j][k]) for k in range(self.shape[2])])
                + "]"
                for j in range(self.shape[1])
            ]
            matrices.append("[" + (",\n " + len(start) * " ").join(matrix_rows) + "]")
        data = (",\n" + len(start) * " ").join(matrices)
        return start + data + "]"

    def __getitem__(self, index: tuple[int, int, int]):
        i, j, k = index
        if i < 0 or i >= self.shape[0]:
            raise IndexError(
                f"index {i} is out of bounds for axis 0 with size {self.shape[0]}"
            )
        if i < 0 or i >= self.shape[0]:
            raise IndexError(
                f"index {j} is out of bounds for axis 1 with size {self.shape[1]}"
            )
        if i < 0 or i >= self.shape[0]:
            raise IndexError(
                f"index {k} is out of bounds for axis 2 with size {self.shape[2]}"
            )
        return self.data[i][j][k]

    def __neg__(self) -> Tensor3D:
        return Tensor3D([[[-x for x in row] for row in matrix] for matrix in self.data])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor3D):
            return NotImplemented
        return self.data == other.data

    def __add__(self, other: Tensor3D) -> Tensor3D:
        if self.shape != other.shape:
            raise ValueError(
                f"tensors of shapes {self.shape} and {other.shape} cannot be added"
            )
        return Tensor3D(
            [
                [
                    [x + y for x, y in zip(row, other_row)]
                    for row, other_row in zip(matrix, other_matrix)
                ]
                for matrix, other_matrix in zip(self.data, other.data)
            ]
        )

    def __sub__(self, other: Tensor3D) -> Tensor3D:
        if self.shape != other.shape:
            raise ValueError(
                f"tensors of shapes {self.shape} and {other.shape} cannot be subtracted"
            )
        return Tensor3D(
            [
                [
                    [x - y for x, y in zip(row, other_row)]
                    for row, other_row in zip(matrix, other_matrix)
                ]
                for matrix, other_matrix in zip(self.data, other.data)
            ]
        )

    def normsqr(self):
        """Compute square of the Euclidean norm of the tensor."""
        return sum(x**2 for submatrix in self.data for row in submatrix for x in row)

    def norm(self):
        """Compute the Euclidean norm of the tensor."""
        return pyo.sqrt(self.normsqr())

    def map(self, f: Callable) -> Tensor3D:
        """Applies map f to each entry of the tensor."""
        return Tensor3D(
            [[[f(x) for x in row] for row in submatrix] for submatrix in self.data]
        )

    def numpy(self) -> npt.NDArray:
        """Return copy of the tensor as numpy array."""
        return np.array(self.data)
