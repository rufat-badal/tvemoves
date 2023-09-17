from math import prod


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
        if i < -self.length or i >= self.length:
            raise IndexError("vector index out of bounds")
        return self._data[i]

    def normsqr(self):
        return sum(x**2 for x in self._data)

    def dot(self, other):
        if self.length != other.length:
            raise ValueError("vectors must have the same length for dot product")
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
            self.rows = [
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
        self.rows = init_list

    def __repr__(self):
        lines = ["[" + ", ".join([repr(x) for x in row]) + "]" for row in self.rows]
        typeinfo = "Matrix(["
        data = (",\n" + len(typeinfo) * " ").join(lines)
        return typeinfo + data + ")]"


M = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print(M)
N = Matrix(list(range(1, 10)), (3, 3))
print(N)
