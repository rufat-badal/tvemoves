from .tensors import Matrix
import pyomo.environ as pyo


def austenite_percentage(theta: float):
    return 1 - 1 / (1 + theta)


def dissipation_potential(dot_C: Matrix):
    return dot_C.normsqr() / 2


def neo_hook(F: Matrix):
    trace_C = (F.transpose() @ F).trace()
    det_F = F.det()
    return trace_C - 2 - 2 * pyo.log(det_F) + (det_F - 1) ** 2


def austenite_potential(F: Matrix):
    return neo_hook(F)


def generate_martensite_potential(scaling_matrix: Matrix):
    return lambda F: austenite_potential(F @ scaling_matrix)


def gradient_austenite_potential(F: Matrix):
    det_F = F.det()
    gradient_det = Matrix([[F[2, 2], -F[2, 1]], [-F[1, 2], F[1, 1]]])
    return 2 * (F + (det_F - 1 / det_F - 1) * gradient_det)


def generate_gradient_martensite_potential(scaling_matrix: Matrix):
    # chain rule
    return (
        lambda F: gradient_austenite_potential(F @ scaling_matrix)
        @ scaling_matrix.transpose()
    )


def internal_energy_weight(theta):
    # a = austenite_percentage
    # a(θ) - θ a'(θ)
    return theta**2 / ((1 + theta) ** 2)


def antider_internal_energy_weight(theta):
    # a = austenite_percentage
    # integral of a(s) - s a'(s) from s = 0 to s = θ
    return (theta * (2 + theta)) / (1 + theta) - 2 * pyo.log(1 + theta)


def symmetrized_strain_delta(prev_F, F):
    delta_F = F - prev_F
    return delta_F.transpose() @ prev_F + prev_F.transpose() @ delta_F


def compose_to_integrand(outer, *inner):
    return lambda *args: outer(*(f(*args) for f in inner))
