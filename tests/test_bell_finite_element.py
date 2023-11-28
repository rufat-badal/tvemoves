"""Test shape function code."""

from tvemoves_rufbad.bell_finite_element import (
    _cyclic_permutation,
    _N_first_third,
    _b,
    _c,
    _L,
)


def test_cyclic_permutation() -> None:
    """Assure that cyclic permutation restores the original expression."""
    n_first_third_new = _N_first_third
    for _ in range(3):
        n_first_third_new = [
            _cyclic_permutation(ni, _L, _b, _c) for ni in n_first_third_new
        ]
    assert all(ni_new == ni for ni_new, ni in zip(n_first_third_new, _N_first_third))
