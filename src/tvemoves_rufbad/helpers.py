"""Module providing the definitions of important potentials and modeling functions."""

import sympy as sp
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from tvemoves_rufbad.tensors import Matrix, inverse_2x2, Tensor3D

ENTROPY_CONSTANT = 10
HEAT_CONDUCTIVITY = Matrix([[1.0, 0.0], [0.0, 1.0]])
HEAT_TRANSFER_COEFFICIENT = 10.0

_theta, _F = sp.symbols("theta F")

_austenite_percentage_symbolic = _theta / (1 + _theta)

austenite_percentage = sp.lambdify([_theta], _austenite_percentage_symbolic)

_internal_energy_weight_symbolic = _austenite_percentage_symbolic - _theta * sp.diff(
    _austenite_percentage_symbolic, _theta
)
internal_energy_weight = sp.lambdify([_theta], _internal_energy_weight_symbolic)

_theta_lim = sp.Symbol("theta_lim")

_antider_internal_energy_weight_symbolic = sp.integrate(
    _internal_energy_weight_symbolic, (_theta, 0, _theta_lim)
)

antider_internal_energy_weight = sp.lambdify(
    [_theta_lim], _antider_internal_energy_weight_symbolic, {"log": pyo.log}
)


def dissipation_potential(prev_strain: Matrix, strain: Matrix):
    """Potential of dissipative forces."""
    return symmetrized_strain_delta(prev_strain, strain).normsqr() / 2


def dissipation_rate(prev_strain: Matrix, strain: Matrix, fps: float):
    """Dissipation rate (twice the dissipation potential)."""
    # remove factor 10
    return dissipation_potential(prev_strain, strain) * 2 * fps**2


def symmetrized_strain_delta(prev_strain: Matrix, strain: Matrix) -> Matrix:
    """Computes discrete symmetrized strain rate given two strains."""
    strain_delta = strain - prev_strain
    return strain_delta.transpose() @ prev_strain + prev_strain.transpose() @ strain_delta


_F_11, _F_12, _F_21, _F_22 = sp.symbols("F_11 F_12 F_21 F_22")
_F = sp.Matrix([[_F_11, _F_12], [_F_21, _F_22]])
_neo_hooke_symbolic = ((_F.T @ _F).trace() / _F.det() - 2) ** 2 + (_F.det() + 1 / _F.det() - 2) ** 4

_neo_hooke_flat_input = sp.lambdify([_F_11, _F_12, _F_21, _F_22], _neo_hooke_symbolic)


def neo_hooke(strain: Matrix):
    """Neo hooke potential. Minimized at SO(d) with required growth rates."""
    return _neo_hooke_flat_input(strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1])


_derivative_neo_hooke_symbolic = [
    [sp.diff(_neo_hooke_symbolic, _F[i, j]) for j in range(2)] for i in range(2)
]
_derivative_neo_hooke_flat_input = sp.lambdify(
    [_F_11, _F_12, _F_21, _F_22], _derivative_neo_hooke_symbolic
)


def derivative_neo_hooke(strain: Matrix) -> Matrix:
    """Gradient of the Neo Hooke potential."""
    return Matrix(
        _derivative_neo_hooke_flat_input(strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1])
    )


def austenite_potential(strain: Matrix):
    """Austenite potential."""
    return neo_hooke(strain)


def derivative_austenite_potential(strain: Matrix):
    """Derivative of the Austenite potential."""
    return derivative_neo_hooke(strain)


FIRST_MARTENSITE_WELL_INVERSE = Matrix([[1.0, -0.5], [0.0, 1.0]])
SECOND_MARTENSITE_WELL_INVERSE = Matrix([[1.0, 0.5], [0.0, 1.0]])


def martensite_potential(strain: Matrix):
    """Martensite potential."""
    return neo_hooke(strain @ FIRST_MARTENSITE_WELL_INVERSE) * neo_hooke(
        strain @ SECOND_MARTENSITE_WELL_INVERSE
    )


def derivative_martensite_potential(strain: Matrix) -> Matrix:
    """Derivative of the Martensite potential."""
    # Product rule
    return (
        derivative_neo_hooke(strain @ FIRST_MARTENSITE_WELL_INVERSE)
        @ FIRST_MARTENSITE_WELL_INVERSE.transpose()
        * neo_hooke(strain @ SECOND_MARTENSITE_WELL_INVERSE)
        + neo_hooke(strain @ FIRST_MARTENSITE_WELL_INVERSE)
        * derivative_neo_hooke(strain @ SECOND_MARTENSITE_WELL_INVERSE)
        @ SECOND_MARTENSITE_WELL_INVERSE.transpose()
    )


def total_elastic_potential(strain: Matrix, theta):
    """Total elastic potential (sum of purely elastic and coupling potential) without entropy."""
    return austenite_percentage(theta) * austenite_potential(strain) + (
        1 - austenite_percentage(theta)
    ) * martensite_potential(strain)


HYPER_ELASTIC_POWER = 3


def hyper_elastic_potential(hyperstrain: Tensor3D):
    """Hyper elastic potential."""
    return hyperstrain.norm() ** HYPER_ELASTIC_POWER


def strain_derivative_coupling_potential(strain: Matrix, theta):
    """F-derivative of the coupling potential."""
    return austenite_percentage(theta) * (
        derivative_austenite_potential(strain) - derivative_martensite_potential(strain)
    )


def internal_energy_no_entropy(strain: Matrix, theta):
    """Internal energy without the entropy term."""
    return internal_energy_weight(theta) * (
        austenite_potential(strain) - martensite_potential(strain)
    )


def internal_energy(strain: Matrix, theta):
    """Internal energy."""
    return internal_energy_no_entropy(strain, theta) + ENTROPY_CONSTANT * theta


def temp_antrider_internal_energy_no_entropy(strain: Matrix, theta):
    """Antiderivative in temperature of the internal energy without the entropy term."""
    return antider_internal_energy_weight(theta) * (
        austenite_potential(strain) - martensite_potential(strain)
    )


def heat_conductivity_reference(strain: Matrix) -> Matrix:
    """Transform heat conductivity tensor into the reference configuration."""
    strain_inverse = inverse_2x2(strain)
    return strain.det() * strain_inverse @ HEAT_CONDUCTIVITY @ strain_inverse.transpose()


def compose_to_integrand(outer, *inner):
    """Compose an admissable integrand 'inner' with a function 'outer' to create a new integrand."""
    return lambda *args: outer(*(f(*args) for f in inner))


def axis():
    """Create a single axis with no ticks and labels."""
    _, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect(1)

    return ax
