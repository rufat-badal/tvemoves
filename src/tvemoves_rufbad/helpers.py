"""Module providing the definitions of important potentials and modeling functions."""

import sympy as sp
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from tvemoves_rufbad.tensors import Matrix, inverse_2x2, Tensor3D

ENTROPY_CONSTANT = 10
HEAT_CONDUCTIVITY = Matrix([[1.0, 0.0], [0.0, 1.0]])
HEAT_TRANSFER_COEFFICIENT = 1
HYPER_ELASTIC_POWER = 3
MARTENSITE_POS_SHEAR = 0.5
FIRST_MARTENSITE_WELL_INVERSE = Matrix([[1.0, -MARTENSITE_POS_SHEAR], [0.0, 1.0]])
SECOND_MARTENSITE_WELL_INVERSE = Matrix([[1.0, MARTENSITE_POS_SHEAR], [0.0, 1.0]])
C1 = 15.037 / 2
D1 = 29.190 / 2


_theta = sp.symbols("theta")

# _austenite_percentage_symbolic = _theta / (1 + _theta)
_austenite_percentage_symbolic = 1 - sp.exp(-5 * _theta)
austenite_percentage = sp.lambdify(
    [_theta], _austenite_percentage_symbolic, modules={"exp": pyo.exp}
)

_internal_energy_weight_symbolic = _austenite_percentage_symbolic - _theta * sp.diff(
    _austenite_percentage_symbolic, _theta
)
internal_energy_weight = sp.lambdify(
    [_theta], _internal_energy_weight_symbolic, modules={"exp": pyo.exp}
)

_theta_lim = sp.Symbol("theta_lim")
_antider_internal_energy_weight_symbolic = sp.integrate(
    _internal_energy_weight_symbolic, (_theta, 0, _theta_lim)
)
antider_internal_energy_weight = sp.lambdify(
    [_theta_lim], _antider_internal_energy_weight_symbolic, modules={"exp": pyo.exp}
)

_F_11, _F_12, _F_21, _F_22 = sp.symbols("F_11 F_12 F_21 F_22")
_F = sp.Matrix([[_F_11, _F_12], [_F_21, _F_22]])

# see https://en.wikipedia.org/wiki/Neo-Hookean_solid
_I1 = (_F.T @ _F).trace()
_J = _F.det()
# alternative definition of the Neo hooke
# _austenite_potential_symbolic = C1 * (_I1 - 2 - 2 * sp.log(_J)) + D1 * (_J - 1) ** 2
_austenite_potential_symbolic = C1 * (_I1 - 2) + (C1 / 6 + D1 / 4) * (_J**2 + 1 / (_J**2) - 2) ** 2
_austenite_potential_flat_input = sp.lambdify(
    [_F_11, _F_12, _F_21, _F_22], _austenite_potential_symbolic, modules=[{"log": pyo.log}]
)

_derivative_austenite_potential_symbolic = [
    [sp.diff(_austenite_potential_symbolic, _F[i, j]) for j in range(2)] for i in range(2)
]
_derivative_austenite_potential_flat_input = sp.lambdify(
    [_F_11, _F_12, _F_21, _F_22], _derivative_austenite_potential_symbolic
)

# Squaring the first term is not standard
# Think about a more physical choice later on
_u = sp.symbols("u")
# alternative double well that has quadratic growth at infinity
# _double_well = _u**2 / (1 + _u)
_double_well = _u**2
_martensite_potential_symbolic = (
    C1 * _double_well.subs(_u, _I1 - 2 - MARTENSITE_POS_SHEAR**2)
    + (C1 / 6 + D1 / 4) * (_J**2 + 1 / (_J**2) - 2) ** 2
)
_martensite_potential_flat_input = sp.lambdify(
    [_F_11, _F_12, _F_21, _F_22], _martensite_potential_symbolic, modules=[{"log": pyo.log}]
)

_derivative_martensite_potential_symbolic = [
    [sp.diff(_martensite_potential_symbolic, _F[i, j]) for j in range(2)] for i in range(2)
]
_derivative_martensite_potential_flat_input = sp.lambdify(
    [_F_11, _F_12, _F_21, _F_22], _derivative_martensite_potential_symbolic
)


def dissipation_potential(prev_strain: Matrix, strain: Matrix):
    """Potential of dissipative forces."""
    return symmetrized_strain_delta(prev_strain, strain).normsqr() / 2


def dissipation_rate(prev_strain: Matrix, strain: Matrix, fps: float):
    """Dissipation rate (twice the dissipation potential)."""
    return dissipation_potential(prev_strain, strain) * 2 * fps**2


def symmetrized_strain_delta(prev_strain: Matrix, strain: Matrix) -> Matrix:
    """Computes discrete symmetrized strain rate given two strains."""
    strain_delta = strain - prev_strain
    return strain_delta.transpose() @ prev_strain + prev_strain.transpose() @ strain_delta


def austenite_potential(strain: Matrix):
    """Austenite potential."""
    return _austenite_potential_flat_input(strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1])


def derivative_austenite_potential(strain: Matrix):
    """Derivative of the Austenite potential."""
    return Matrix(
        _derivative_austenite_potential_flat_input(
            strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1]
        )
    )


def martensite_potential(strain: Matrix):
    """Martensite potential."""
    return _martensite_potential_flat_input(strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1])


def derivative_martensite_potential(strain: Matrix) -> Matrix:
    """Derivative of the Martensite potential."""
    # Product rule
    return Matrix(
        _derivative_martensite_potential_flat_input(
            strain[0, 0], strain[0, 1], strain[1, 0], strain[1, 1]
        )
    )


def total_elastic_potential(strain: Matrix, theta):
    """Total elastic potential (sum of purely elastic and coupling potential) without entropy."""
    return austenite_percentage(theta) * austenite_potential(strain) + (
        1 - austenite_percentage(theta)
    ) * martensite_potential(strain)


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


def fig_axis(xlims: tuple[float, float], ylims: tuple[float, float]):
    """Create a single axis with no ticks and labels."""
    fig, ax = plt.subplots()
    setup_axis(ax, xlims, ylims)
    return fig, ax


def setup_axis(ax, xlims: tuple[float, float], ylims: tuple[float, float]):
    ax.axis("off")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect(1)
