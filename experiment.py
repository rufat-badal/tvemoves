from tvemoves_rufbad.bell_finite_element import *

for Ni_left, Ni_right in zip(
    shape_function_on_segment_symbolic_left, shape_function_on_segment_symbolic_right
):
    print(Ni_left)
    print(Ni_right)
    print()
