from __future__ import annotations
from itertools import permutations
from typing import Callable


def sign(p):
    # p must be a permutation of [0, 1, 2, ...]
    num_misplaced = 0
    for i, a in enumerate(p):
        for b in p[i + 1 :]:
            if b < a:
                num_misplaced += 1

    return 1 if num_misplaced % 2 == 0 else -1


class Vector:
    def __init__(self, data: list):
        self.shape = (len(data),)
        self._data = data

    def __repr__(self) -> str:
        return (
            "Vector(["
            + ", ".join([str(self._data[i]) for i in range(self.shape[0])])
            + "])"
        )

    def __getitem__(self, i):
        # do not allow negative indices
        if i < 0 or i >= self.shape[0]:
            raise IndexError("vector index out of bounds")
        return self._data[i]

    def __neg__(self) -> Vector:
        return Vector([-x for x in self._data])

    def __add__(self, other: Vector) -> Vector:
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for addition")
        return Vector([x + y for x, y in zip(self._data, other._data)])

    def __sub__(self, other: Vector) -> Vector:
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for subtraction")
        return Vector([x - y for x, y in zip(self._data, other._data)])

    def normsqr(self):
        return sum(x**2 for x in self._data)

    def dot(self, other: Vector):
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for the dot product")
        return sum(x * y for (x, y) in zip(self._data, other._data))

    def __mul__(self, scaling) -> Vector:
        return Vector([x * scaling for x in self._data])

    def __rmul__(self, scaling) -> Vector:
        return self.__mul__(scaling)

    def __truediv__(self, divisor) -> Vector:
        return Vector([x / divisor for x in self._data])

    def __rtruediv__(self, divisor):
        return NotImplemented

    def reshape(self, num_rows: int, num_cols: int) -> Matrix:
        # row major format
        return Matrix(
            [
                [self._data[i * num_cols + j] for j in range(num_cols)]
                for i in range(num_rows)
            ]
        )

    def vstack(self, other: Vector) -> Matrix:
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                "vectors must be of the same length to be stacked vertically"
            )
        return Matrix(
            [
                self._data,
                other._data,
            ]
        )

    def map(self, f: Callable) -> Vector:
        return Vector([f(x) for x in self._data])


class Matrix:
    def __init__(self, data: list[list]):
        # row major format
        self.shape = (len(data), len(data[0]))
        for row in data[1:]:
            if len(row) != self.shape[1]:
                raise ValueError("incorrectly shaped initialization list provided")
        self._data = data

    def __repr__(self):
        lines = [
            "["
            + ", ".join([repr(self._data[i][j]) for j in range(self.shape[1])])
            + "]"
            for i in range(self.shape[0])
        ]
        typeinfo = "Matrix(["
        data = (",\n" + len(typeinfo) * " ").join(lines)
        return typeinfo + data + ")]"

    def __getitem__(self, index: tuple[int, int]):
        i, j = index
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            raise IndexError("matrix index out of bounds")
        return self._data[i][j]

    def __neg__(self) -> Matrix:
        return Matrix([[-x for x in row] for row in self._data])

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for addition")
        return Matrix(
            [
                [x + y for (x, y) in zip(row, other_row)]
                for (row, other_row) in zip(self._data, other._data)
            ]
        )

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for subtraction")
        return Matrix(
            [
                [x - y for (x, y) in zip(row, other_row)]
                for (row, other_row) in zip(self._data, other._data)
            ]
        )

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.shape[1] != other.shape[0]:
            raise ValueError("matrix shapes do not match for multiplication")
        return Matrix(
            [
                [
                    sum(
                        self._data[i][k] * other._data[k][j]
                        for k in range(self.shape[1])
                    )
                    for j in range(other.shape[1])
                ]
                for i in range(self.shape[0])
            ]
        )

    def transpose(self) -> Matrix:
        return Matrix(
            [
                [self._data[i][j] for i in range(self.shape[0])]
                for j in range(self.shape[1])
            ]
        )

    def trace(self):
        return sum(self._data[i][i] for i in range(min(self.shape)))

    def det(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("det not defined for nonsquare matrix")
        res = 0
        for p in permutations(range(self.shape[0])):
            prod = 1
            for i in range(self.shape[0]):
                prod *= self._data[i][p[i]]
            res += sign(p) * prod

        return res

    def normsqr(self):
        return sum(x**2 for row in self._data for x in row)

    def scalar_product(self, other):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same length for the dot product")
        return sum(
            x * y
            for (row, other_row) in zip(self._data, other._data)
            for (x, y) in zip(row, other_row)
        )

    def __mul__(self, scaling) -> Matrix:
        return Matrix([[x * scaling for x in row] for row in self._data])

    def __rmul__(self, scaling) -> Matrix:
        return self.__mul__(scaling)

    def __truediv__(self, divisor) -> Matrix:
        return Matrix([[x / divisor for x in row] for row in self._data])

    def __rtruediv__(self, divisor):
        return NotImplemented

    def flatten(self) -> Vector:
        # row major format
        return Vector([x for row in self._data for x in row])

    def dot(self, v) -> Vector:
        if v.shape[0] != self.shape[1]:
            raise ValueError(
                "shape of matrix and vector do not match for the matrix-vector-product"
            )
        return Vector(
            [
                sum(self._data[i][j] * v[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ]
        )

    def map(self, f: Callable) -> Matrix:
        return Matrix([[f(x) for x in row] for row in self._data])

    def vstack(self, other: Matrix) -> Tensor3D:
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                "matrices must be of the same length to be stacked vertically"
            )
        return Tensor3D(
            [
                self._data,
                other._data,
            ]
        )


class Tensor3D:
    def __init__(self, data: list[list[list]]):
        self.shape = (len(data), len(data[0]), len(data[0][0]))
        for submatrix in data:
            if len(submatrix) != self.shape[1]:
                raise ValueError("incorrectly shaped initialization list provided")
            for row in submatrix:
                if len(row) != self.shape[2]:
                    raise ValueError("incorrectly shaped initialization list provided")
        self._data = data

    def __repr__(self):
        typeinfo = "Tensor3D(["
        matrices_representations = []
        for i in range(self.shape[0]):
            matrix_rows = [
                "["
                + ", ".join([repr(self._data[i][j][k]) for k in range(self.shape[2])])
                + "]"
                for j in range(self.shape[1])
            ]
            matrices_representations.append(
                "[" + (",\n " + len(typeinfo) * " ").join(matrix_rows) + "]"
            )
        data = (",\n" + len(typeinfo) * " ").join(matrices_representations)
        return typeinfo + data + ")]"

    def __getitem__(self, index: tuple[int, int, int]):
        i, j, k = index
        if (
            i < 0
            or i >= self.shape[0]
            or j < 0
            or j >= self.shape[1]
            or k < 0
            or k >= self.shape[2]
        ):
            raise IndexError("tensor index out of bounds")
        return self._data[i][j][k]

    def normsqr(self):
        return sum(x**2 for submatrix in self._data for row in submatrix for x in row)

    def map(self, f: Callable) -> Tensor3D:
        return Tensor3D(
            [[[f(x) for x in row] for row in submatrix] for submatrix in self._data]
        )
