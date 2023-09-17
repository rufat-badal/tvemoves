from math import prod
from itertools import permutations


class Vector:
    def __init__(self, init_list: list):
        self._data = init_list
        self.length = len(init_list)

    def __repr__(self):
        return "Vector([" + ", ".join([str(x) for x in self._data]) + "])"

    def __add__(self, other):
        if self.length != other.length:
            raise ValueError("vectors must have the same length for addition")
        return Vector([x + y for (x, y) in zip(self._data, other._data)])

    def __sub__(self, other):
        if self.length != other.length:
            raise ValueError("vectors must have the same length for subtraction")
        return Vector([x - y for (x, y) in zip(self._data, other._data)])

    def __neg__(self):
        return Vector([-x for x in self._data])

    def __getitem__(self, i):
        # do not allow negative indices
        if i < 0 or i >= self.length:
            raise IndexError("vector index out of bounds")
        return self._data[i]

    def normsqr(self):
        return sum(x**2 for x in self._data)

    def dot(self, other):
        if self.length != other.length:
            raise ValueError("vectors must have the same length for the dot product")
        return sum(x * y for (x, y) in zip(self._data, other._data))


class Matrix:
    def __init__(self, init_list: list, shape=None):
        if type(init_list[0]) is not list:
            if shape is not None and shape[0] * shape[1] != len(init_list):
                raise ValueError("shape is not compatible with the data provided")
            if shape is None:
                shape = (1, len(init_list))
            self.shape = shape
            self._data = init_list
            self._unflattened = [
                self._data[i : i + self.shape[1]]
                for i in range(0, self.shape[0] * self.shape[1], self.shape[1])
            ]
            return

        # row major format
        num_rows = len(init_list)
        num_columns = len(init_list[0])
        for row in init_list[1:]:
            if len(row) != num_columns:
                raise ValueError("rows must have all the same length")
        self.shape = (num_rows, num_columns)
        self._data = [x for row in init_list for x in row]
        self._unflattened = init_list

    def __repr__(self):
        lines = [
            "[" + ", ".join([repr(x) for x in row]) + "]" for row in self._unflattened
        ]
        typeinfo = "Matrix(["
        data = (",\n" + len(typeinfo) * " ").join(lines)
        return typeinfo + data + ")]"

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for addition")
        return Matrix([x + y for (x, y) in zip(self._data, other._data)], self.shape)

    def __sub__(self, other):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same shape for subtraction")
        return Matrix([x - y for (x, y) in zip(self._data, other._data)], self.shape)

    def __neg__(self):
        return Matrix([-x for x in self._data], self.shape)

    def __getitem__(self, index):
        i, j = index
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            raise IndexError("matrix index out of bounds")
        return self._data[i * self.shape[1] + j]

    def __matmul__(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("matrix shapes do not match for multiplication")
        return Matrix(
            [
                [
                    sum(self[i, k] * other[k, j] for k in range(self.shape[1]))
                    for j in range(self.shape[1])
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
        return sum(x**2 for x in self._data)

    def dot(self, other):
        if self.shape != other.shape:
            raise ValueError("matrices must have the same length for the dot product")
        return sum(x * y for (x, y) in zip(self._data, other._data))

    def __mul__(self, scaling):
        return Matrix(
            [[scaling * self[i, j] for j in self.shape[1]] for i in self.shape[0]]
        )

    def __rmul__(self, scaling):
        return self.__mul__(scaling)


def sign(p):
    # p must be a permutation of [0, 1, 2, ...]
    num_misplaced = 0
    for i, a in enumerate(p):
        for b in p[i + 1 :]:
            if b < a:
                num_misplaced += 1

    return 1 if num_misplaced % 2 == 0 else -1


class Tensor3D:
    def __init__(self, init_list: list, shape=None):
        if type(init_list[0]) is not list:
            if shape is not None and shape[0] * shape[1] * shape[2] != len(init_list):
                raise ValueError("shape is not compatible with the data provided")
            if shape is None:
                shape = (1, 1, len(init_list))
            self.shape = shape
            self._data = init_list
            self._unflattened = [
                [
                    self._data[i + j : i + j + self.shape[2]]
                    for i in range(0, self.shape[1] * self.shape[2], self.shape[2])
                ]
                for j in range(
                    0,
                    self.shape[0] * self.shape[1] * self.shape[2],
                    self.shape[1] * self.shape[2],
                )
            ]
            return

        if type(init_list[0][0]) is not list:
            raise ValueError("incorrectly shaped initialization list provided")
        self.shape = (len(init_list), len(init_list[0]), len(init_list[0][0]))
        for submatrix in init_list:
            if len(submatrix) != self.shape[1]:
                raise ValueError("incorrectly shaped initialization list provided")
            for row in submatrix:
                if len(row) != self.shape[2]:
                    raise ValueError("incorrectly shaped initialization list provided")
        self._data = [x for submatrix in init_list for row in submatrix for x in row]
        self._unflattened = init_list


A = Tensor3D(list(range(1, 25)), (2, 3, 4))
print(A._unflattened)
B = Tensor3D(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)
print(B.shape)
print(B._data)
print(B._unflattened)
