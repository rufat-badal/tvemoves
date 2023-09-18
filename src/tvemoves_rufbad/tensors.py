from __future__ import annotations
from itertools import permutations


class Vector:
    def __init__(self, data: list):
        self.shape = (len(data),)
        self._data = data

    def __repr__(self):
        return "Vector([" + ", ".join([str(x) for x in self._data]) + "])"

    def __add__(self, other: Vector):
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for addition")
        return Vector([x + y for (x, y) in zip(self._data, other._data)])

    def __sub__(self, other: Vector):
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for subtraction")
        return Vector([x - y for (x, y) in zip(self._data, other._data)])

    def __neg__(self):
        return Vector([-x for x in self._data])

    def __getitem__(self, i: int):
        # do not allow negative indices
        if i < 0 or i >= self.shape[0]:
            raise IndexError("vector index out of bounds")
        return self._data[i]

    def __mul__(self, scaling):
        return Vector([scaling * self[i] for i in range(self.shape[0])])

    def __rmul__(self, scaling):
        return self.__mul__(scaling)

    def normsqr(self):
        return sum(x**2 for x in self._data)

    def dot(self, other: Vector):
        if self.shape != other.shape:
            raise ValueError("vectors must have the same length for the dot product")
        return sum(x * y for (x, y) in zip(self._data, other._data))


class Matrix:
    def __init__(self, data: list[list]):
        # row major format
        self.shape = (len(data), len(data[0]))
        for row in data[1:]:
            if len(row) != self.shape[1]:
                raise ValueError("incorrectly shaped initialization list provided")
        self._data = data

    def __repr__(self):
        lines = ["[" + ", ".join([repr(x) for x in row]) + "]" for row in self._data]
        typeinfo = "Matrix(["
        data = (",\n" + len(typeinfo) * " ").join(lines)
        return typeinfo + data + ")]"

    def __add__(self, other: Matrix):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for addition")
        return Matrix(
            [
                [x + y for (x, y) in zip(row, other_row)]
                for (row, other_row) in zip(self._data, other._data)
            ]
        )

    def __sub__(self, other: Matrix):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for subtraction")
        return Matrix(
            [
                [x - y for (x, y) in zip(row, other_row)]
                for (row, other_row) in zip(self._data, other._data)
            ]
        )

    def __neg__(self):
        return Matrix([[-x for x in row] for row in self._data])

    def __getitem__(self, index: tuple[int, int]):
        i, j = index
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            raise IndexError("matrix index out of bounds")
        return self._data[i][j]

    def __matmul__(self, other: Matrix):
        if self.shape[1] != other.shape[0]:
            raise ValueError("matrix shapes do not match for multiplication")
        return Matrix(
            [
                [
                    sum(self[i, k] * other[k, j] for k in range(self.shape[1]))
                    for j in range(other.shape[1])
                ]
                for i in range(self.shape[0])
            ]
        )

    @property
    def T(self):
        return Matrix(
            [[self[i, j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        )

    def trace(self):
        return sum(self[i, i] for i in range(min(self.shape)))

    def det(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("det not defined for nonsquare matrix")
        res = 0
        for p in permutations(range(self.shape[0])):
            prod = 1
            for i in range(self.shape[0]):
                prod *= self[i, p[i]]
            res += sign(p) * prod

        return res

    def normsqr(self):
        return sum(x**2 for row in self._data for x in row)

    def dot(self, other: Matrix):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same length for the dot product")
        return sum(
            x * y
            for (row, other_row) in zip(self._data, other._data)
            for (x, y) in zip(row, other_row)
        )

    def __mul__(self, scaling):
        return Matrix(
            [
                [scaling * self[i, j] for j in range(self.shape[1])]
                for i in range(self.shape[0])
            ]
        )

    def __rmul__(self, scaling):
        return self.__mul__(scaling)


def sign(p: list[int]):
    # p must be a permutation of [0, 1, 2, ...]
    num_misplaced = 0
    for i, a in enumerate(p):
        for b in p[i + 1 :]:
            if b < a:
                num_misplaced += 1

    return 1 if num_misplaced % 2 == 0 else -1


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

    def normsqr(self):
        return sum(x**2 for submatrix in self._data for row in submatrix for x in row)
