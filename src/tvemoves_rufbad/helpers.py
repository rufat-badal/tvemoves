"""Module providing the definitions of important potentials and modeling functions."""

import pyomo.environ as pyo
from tvemoves_rufbad.tensors import Matrix


def austenite_percentage(theta):
    """Percentage of austenite in a material of temperature theta."""
    return 1 - 1 / (1 + theta)


def dissipation_norm(symmetrized_strain_rate: Matrix):
    """Norm of a symmetrized strain rate."""
    return symmetrized_strain_rate.normsqr() / 2


def symmetrized_strain_delta(prev_strain, strain):
    """Computes discrete symmetrized strain rate given two strains."""
    strain_delta = strain - prev_strain
    return strain_delta.transpose() @ prev_strain + prev_strain.transpose() @ strain_delta


def dissipation_potential(prev_strain, strain):
    """Dissipation generated going from prev_strain to strain."""
    return dissipation_norm(symmetrized_strain_delta(prev_strain, strain))


def neo_hook(strain: Matrix):
    """Neo hook potential."""
    trace_of_symmetrized_strain = (strain.transpose() @ strain).trace()
    det_of_strain = strain.det()
    return trace_of_symmetrized_strain - 2 - 2 * pyo.log(det_of_strain) + (det_of_strain - 1) ** 2


def austenite_potential(strain: Matrix):
    """Austenite potential."""
    return neo_hook(strain)


def create_martensite_potential(scaling_matrix: Matrix):
    """Create martensite potential for a given scaling matrix."""
    return lambda strain: austenite_potential(strain @ scaling_matrix)


def gradient_austenite_potential(strain: Matrix):
    """Gradient of the austenite potential."""
    det_of_strain = strain.det()
    gradient_of_det = Matrix([[strain[2, 2], -strain[2, 1]], [-strain[1, 2], strain[1, 1]]])
    return 2 * (strain + (det_of_strain - 1 / det_of_strain - 1) * gradient_of_det)


def gradient_martensite_potential(scaling_matrix: Matrix):
    """Create gradient of the martensite potential for a given scaling matrix."""
    # chain rule
    return lambda F: gradient_austenite_potential(F @ scaling_matrix) @ scaling_matrix.transpose()


def internal_energy_weight(theta):
    """a(θ) - θ a'(θ) where a(θ) is the austenite percentage"""
    return theta**2 / ((1 + theta) ** 2)


def antider_internal_energy_weight(theta):
    """Integral of a(s) - s a'(s) from s = 0 to s = θ"""
    return (theta * (2 + theta)) / (1 + theta) - 2 * pyo.log(1 + theta)


def compose_to_integrand(outer, *inner):
    """Compose an admissable integrand 'inner' with a function 'outer' to create a new integrand."""
    return lambda *args: outer(*(f(*args) for f in inner))
