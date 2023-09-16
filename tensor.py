from math import prod


class Tensor:
    def __init__(self, shape):
        self._num_entries = prod(shape)
        self.shape = shape
        self._data = [None for _ in range(self._num_entries)]


t = Tensor((2, 3, 4))
print(t._data)


# def __str__(self):
#     start = "(["
#     lines = []
#     for row in self._M:
#         lines.append("[" + ", ".join([str(x) for x in row]) + "]")
#     return start + ("\n" + len(start) * " ").join(lines) + ")]"

# def __repr__(self):
#     typeinfo = "Matrix(["
#     lines = []
#     for row in self._M:
#         lines.append("[" + ", ".join([repr(x) for x in row]) + "]")
#     return typeinfo + ("\n" + len(typeinfo) * " ").join(lines) + ")]"

# def __add__(self, other):
#     assert (
#         self.shape == other.shape
#     ), "Matrices must have the same shape for addition"
#     sum_M = []
#     for i in range(self.num_rows):
#         row = []
#         for j in range(self.num_cols):
#             row.append(self._M[i][j] + other._M[i][j])
#         sum_M.append(row)

#     return Matrix(sum_M)

# def __sub__(self, other):
#     assert (
#         self.shape == other.shape
#     ), "Matrices must have the same shape for subtraction"
#     sub_M = []
#     for i in range(self.num_rows):
#         row = []
#         for j in range(self.num_cols):
#             row.append(self._M[i][j] - other._M[i][j])
#         sub_M.append(row)

#     return Matrix(sub_M)

# def __neg__(self):
#     neg_M = []

#     for i in range(self.num_rows):
#         row = []
#         for j in range(self.num_cols):
#             row.append(-self._M[i][j])
#         neg_M.append(row)

#     return Matrix(neg_M)

# def __matmul__(self, other):
#     assert (
#         self.num_cols == other.num_rows
#     ), "self.num_cols and other.num_rows must coincide for matrix multiplication"
#     prod_M = []
#     for i in range(self.num_rows):
#         row = []
#         for j in range(other.num_cols):
#             row.append(
#                 sum(self._M[i][k] * other._M[k][j] for k in range(self.num_cols))
#             )
#         prod_M.append(row)

#     return Matrix(prod_M)

# @property
# def T(self):
#     transp_M = []
#     for i in range(self.num_rows):
#         row = []
#         for j in range(self.num_cols):
#             row.append(self._M[j][i])
#         transp_M.append(row)

#     return Matrix(transp_M)

# def __getitem__(self, index):
#     i, j = index  # Indices start at 1!
#     i -= 1
#     j -= 1
#     assert (
#         i >= 0 and i < self.num_rows and j >= 0 and j < self.num_cols
#     ), "Index out of bounds"
#     return self._M[i][j]

# def trace(self):
#     assert self.num_cols == self.num_rows, "Not a square matrix"
#     return sum(self._M[i][i] for i in range(self.num_rows))

# def det(self):
#     assert self.num_cols == self.num_rows, "Not a square matrix"
#     n = self.num_cols

#     det = 0
#     for p in permutations(range(n)):
#         prod = 1
#         for i in range(n):
#             prod *= self._M[i][p[i]]
#         det += sign(p) * prod

#     return det

# def normsqr(self):
#     return sum(x**2 for row in self._M for x in row)

# def dot(self, other):
#     assert (
#         self.shape == other.shape
#     ), "Matrices must have the same shape for the dot product"
#     res = 0
#     for (self_row, other_row) in zip(self._M, other._M):
#         for (self_entry, other_entry) in zip(self_row, other_row):
#             res += self_entry * other_entry

#     return res

# def __mul__(self, scaling):
#     scaled_M = []
#     for i in range(self.num_rows):
#         row = []
#         for j in range(self.num_cols):
#             row.append(scaling * self._M[i][j])
#         scaled_M.append(row)

#     return Matrix(scaled_M)

# def __rmul__(self, scaling):
#     return self.__mul__(scaling)
