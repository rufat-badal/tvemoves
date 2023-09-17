from __future__ import annotations
from math import prod


class Vector:
    def __init__(self, init_list: list[float]):
        self._data = init_list
        self._length = len(init_list)

    def __str__(self):
        return "Vector([" + ", ".join([str(x) for x in self._data]) + "])"

    def __add__(self, other: Vector):
        if self._length != other._length:
            raise ValueError("vectors must have the same length for addition")
        return Vector([x + y for (x, y) in zip(self._data, other._data)])

    def __sub__(self, other: Vector):
        if self._length != other._length:
            raise ValueError("vectors must have the same length for subtraction")
        return Vector([x - y for (x, y) in zip(self._data, other._data)])

    def __neg__(self):
        return Vector([-x for x in self._data])

    def __getitem__(self, i):
        if i < -self._length or i >= self._length:
            raise (IndexError("vector index out of bounds"))
        return self._data[i]

    def normsqr(self):
        return sum(x**2 for x in self._data)

    def dot(self, other):
        if self._length != other._length:
            raise ValueError("vectors must have the same length for dot product")
        return sum(x * y for (x, y) in zip(self._data, other._data))


v = Vector([1, 2, 3])
w = Vector([1, 1, 1])
print(v + w)
print(v - w)
print(-v)
print(v[-3])
print(w.normsqr())
print(w)
print(w.dot(w))
