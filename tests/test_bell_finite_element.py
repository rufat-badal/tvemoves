"""Test shape function code."""

from tvemoves_rufbad.bell_finite_element import (
    _cyclic_permutation,
    _N_first_third,
    _N,
    _N_on_edge,
    _b,
    _c,
    _L,
    _t,
)


def test_cyclic_permutation() -> None:
    """Assure that cyclic permutation restores the original expression."""
    n_first_third_new = _N_first_third
    for _ in range(3):
        n_first_third_new = [
            _cyclic_permutation(ni, _L, _b, _c) for ni in n_first_third_new
        ]
    assert all(ni_new == ni for ni_new, ni in zip(n_first_third_new, _N_first_third))


def test_independence_of_opposite_point() -> None:
    """Check that the shape function on an edge is independent of the point opposite to the edge."""
    for i in range(12, 18):
        assert _N[i].subs(_L[2], 0) == 0

    for i in range(12):
        assert _N[i].subs(_L[2], 0).free_symbols.issubset({_L[0], _L[1], _b[2], _c[2]})

    for ni in _N_on_edge:
        assert ni.free_symbols.issubset({_t, _b[2], _c[2]})
