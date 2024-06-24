"""Module provind the simulation class."""

from dataclasses import dataclass
from abc import ABC
import numpy.typing as npt
import numpy as np
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from typing import Callable
from tvemoves_rufbad.domain import Domain, Grid, RefinedGrid
from tvemoves_rufbad.mechanical_step import (
    MechanicalStepParams,
    AbstractMechanicalStep,
    mechanical_step,
)
from tvemoves_rufbad.thermal_step import ThermalStepParams, AbstractThermalStep, thermal_step
from tvemoves_rufbad.interpolation import (
    EuclideanDeformation,
    P1Interpolation,
    C1Interpolation,
    EuclideanInterpolation,
)
from tvemoves_rufbad.helpers import fig_axis, setup_axis

PLOTTING_REFINEMENT_FACTOR = 5


class AbstractStep(ABC):
    """Abstract simulation step"""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_deta: npt.NDArray[np.float64],
        y: EuclideanDeformation,
        theta: EuclideanInterpolation,
        domain: Domain,
        grid: Grid,
    ):
        self._domain_curves = domain.curves
        self._grid = grid
        self._y_data = y_data
        self._theta_data = theta_deta
        self.y = y
        self.theta = theta

    def __repr__(self):
        return f"Step(y={repr(self._y_data)}, theta={repr(self._theta_data)})"

    def __str__(self):
        return f"y:\n{str(self._y_data)}\ntheta:\n{str(self._theta_data)}"

    def _plot_temperature(self, ax, max_temp: float) -> None:
        deformed_points = [self.y(*p) for p in self._grid.points]
        x = [p[0] for p in deformed_points]
        y = [p[1] for p in deformed_points]
        c = [self.theta(*p) for p in self._grid.points]
        ax.tripcolor(
            x,
            y,
            c,
            triangles=self._grid.triangles,
            vmin=0.0,
            vmax=max_temp,
            cmap="plasma",
            # shading="gouraud",
            edgecolors="black",
        )

    def _plot_deformation_curves(
        self,
        num_points_per_curve: int,
        num_horizontal_curves: int = 0,
        num_vertical_curves: int | None = None,
    ):
        for curve in self._domain_curves(
            num_points_per_curve, num_horizontal_curves, num_vertical_curves
        ):
            deformed_curve = [self.y(*p) for p in curve]
            x = [p[0] for p in deformed_curve]
            y = [p[1] for p in deformed_curve]
            plt.plot(x, y, color="gray", linewidth=0.5)

    def plot(
        self,
        ax,
        max_temp: float,
    ):
        self._plot_temperature(ax, max_temp)


class Step(AbstractStep):
    """Not regularized simulation step."""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_data: npt.NDArray[np.float64],
        domain: Domain,
        grid: Grid,
    ):
        if y_data.shape != (len(grid.vertices), 2):
            raise ValueError(f"incorrectly shaped y_data of shape = {y_data.shape} provided")

        if theta_data.shape != (len(grid.vertices),):
            raise ValueError(
                f"incorrectly shaped theta_data provided of shape = {theta_data.shape} provided"
            )

        y1_params = y_data[:, 0].tolist()
        y1_interpolation = P1Interpolation(grid, y1_params)
        y2_params = y_data[:, 1].tolist()
        y2_interpolation = P1Interpolation(grid, y2_params)
        y = EuclideanDeformation(y1_interpolation, y2_interpolation)

        theta = EuclideanInterpolation(P1Interpolation(grid, theta_data.tolist()))

        super().__init__(y_data, theta_data, y, theta, domain, grid)


class RegularizedStep(AbstractStep):
    """Not regularized simulation step."""

    def __init__(
        self,
        y_data: npt.NDArray[np.float64],
        theta_data: npt.NDArray[np.float64],
        domain: Domain,
        grid: Grid,
        refined_grid: RefinedGrid | None,
    ):
        if refined_grid is None:
            raise ValueError("refined grid is required in the regularized setting")
        # last dimension correspond to the degrees of freedom of the C1 interpolation
        if y_data.shape != (len(grid.vertices), 2, 6):
            raise ValueError(
                f"incorrectly shaped y_data of shape = {y_data.shape} provided (should be"
                f" {(len(grid.vertices), 2, 6)})"
            )

        if theta_data.shape != (len(refined_grid.vertices),):
            raise ValueError(
                f"incorrectly shaped theta_data provided of shape = {theta_data.shape} provided"
                f" (should be {(len(refined_grid.vertices),)})"
            )

        y1_params = y_data[:, 0, :].tolist()
        y1_interpolation = C1Interpolation(grid, y1_params)
        y2_params = y_data[:, 1, :].tolist()
        y2_interpolation = C1Interpolation(grid, y2_params)
        y = EuclideanDeformation(y1_interpolation, y2_interpolation)

        theta = EuclideanInterpolation(P1Interpolation(refined_grid, theta_data.tolist()))

        plotting_grid = domain.refine(refined_grid, PLOTTING_REFINEMENT_FACTOR)

        super().__init__(y_data, theta_data, y, theta, domain, plotting_grid)


@dataclass
class SimulationParams:
    """Parameters specifying the simulation."""

    initial_temperature: float
    search_radius: float
    fps: float
    scale: float
    regularization: float | None = None
    refinement_factor: int | None = None

    def __post_init__(self):
        if self.initial_temperature < 0:
            raise ValueError(
                f"initial temperature must be non-negative but {self.initial_temperature} was"
                " provided"
            )

        if self.search_radius <= 0:
            raise ValueError(
                f"search radius must be positive but {self.search_radius} was provided"
            )

        if self.fps < 0:
            raise ValueError(f"fps must be positive but {self.fps} was provided")

        if self.regularization is not None and self.regularization < 0:
            raise ValueError(
                f"regularization must be non-negative but {self.regularization} was provided"
            )

        if self.scale <= 0:
            raise ValueError(f"scale must be positive but {self.scale} was provided")

        if self.refinement_factor is not None and self.refinement_factor <= 0:
            raise ValueError(f"fps must be positive but {self.refinement_factor} was provided")

        if self.regularization is not None and self.refinement_factor is None:
            raise ValueError("refinement_factor cannot be None in the regularized setting")

        if self.regularization is None and self.refinement_factor is not None:
            self.refinement_factor = None

    def mechanical_step_params(
        self,
    ) -> MechanicalStepParams:
        """Return subset of the parameters needed for the mechanical step only."""

        return MechanicalStepParams(
            self.initial_temperature,
            self.search_radius,
            self.fps,
            self.regularization,
        )

    def thermal_step_params(
        self,
    ) -> ThermalStepParams:
        """Return subset of the parameters needed for the mechanical step only."""

        return ThermalStepParams(
            self.search_radius,
            self.fps,
            self.regularization,
        )


class Simulation:
    """Class implementing the minimizing movement scheme with or without regularization."""

    def __init__(
        self,
        domain: Domain,
        params: SimulationParams,
        external_temperature: Callable[[float], float] = lambda t: 0.0,
        boundary_traction: Callable[[float, float, float], list[float]] = lambda t, x, y: [
            0.0,
            0.0,
        ],
    ):
        self._domain = domain
        self._solver = pyo.SolverFactory("ipopt")
        self.params = params
        self._grid = self._domain.grid(self.params.scale)
        self._refined_grid = (
            self._domain.refine(self._grid, self.params.refinement_factor)
            if self.params.refinement_factor is not None
            else None
        )

        # Init and solve first mechanical step
        self._mechanical_step: AbstractMechanicalStep = mechanical_step(
            self._solver,
            self._grid,
            self.params.mechanical_step_params(),
            self._refined_grid,
            boundary_traction,
        )
        self._mechanical_step.solve()

        # Add initial step
        self.steps: list[AbstractStep] = []
        self._tau = 1 / self.params.fps
        self._current_time = -self._tau
        self._max_temp = 0
        self._xlims = (float("inf"), float("-inf"))
        self._ylims = (float("inf"), float("-inf"))
        self._append_step(self._mechanical_step.prev_y(), self._mechanical_step.prev_theta())

        # Init and solve first thermal step
        self._thermal_step: AbstractThermalStep = thermal_step(
            self._solver,
            self._grid,
            self._mechanical_step.prev_y(),
            self._mechanical_step.y(),
            self._mechanical_step.prev_theta(),
            self.params.thermal_step_params(),
            external_temperature,
            self._refined_grid,
        )
        self._thermal_step.solve()
        self._append_step(self._thermal_step.y(), self._thermal_step.theta())

    def max_temp(self):
        return self._max_temp

    def _plot_step(self, i: int):
        if i < -len(self.steps) or i >= len(self.steps):
            return None
        fig, ax = fig_axis(self._xlims, self._ylims)
        self.steps[i].plot(
            ax,
            self.max_temp(),
        )
        return fig

    def plot_step(self, i: int):
        self._plot_step(i)

    def save_step(self, i: int, path: str = "step"):
        fig = self._plot_step(i)
        if fig is not None:
            step_index = i if i >= 0 else len(self.steps) + i
            fig.savefig(
                f"{path}_{step_index}.png",
                bbox_inches="tight",
                transparent="True",
                pad_inches=0,
                dpi=300,
            )

    def save_animation(self, filename: str = "animation"):
        fig, ax = fig_axis(self._xlims, self._ylims)

        def animate_frame(i):
            ax.clear()
            setup_axis(ax, self._xlims, self._ylims)
            self.steps[i].plot(ax, self._max_temp)

        anim = animation.FuncAnimation(fig, animate_frame, frames=len(self.steps))
        anim.save(filename + ".mp4", fps=30, dpi=300, extra_args=["-vcodec", "libx264"])

    def _update_mechanical_step(self):
        self._mechanical_step.update_prev_y(self._thermal_step.y())
        self._mechanical_step.update_prev_theta(self._thermal_step.theta())
        self._mechanical_step.update_boundary_traction(self._current_time)

    def _update_thermal_step(self):
        self._thermal_step.update_prev_y(self._mechanical_step.prev_y())
        self._thermal_step.update_prev_theta(self._mechanical_step.prev_theta())
        self._thermal_step.update_y(self._mechanical_step.y())
        self._thermal_step.update_external_temperature(self._current_time)

    def _append_step(self, y_data: npt.NDArray[np.float64], theta_data: npt.NDArray[np.float64]):
        # Update variables required for plotting
        self._max_temp = max(self._max_temp, np.max(theta_data))
        x_coords = y_data[:, 0]
        y_coords = y_data[:, 1]
        self._xlims = (min(self._xlims[0], np.min(x_coords)), max(self._xlims[1], np.max(x_coords)))
        self._ylims = (min(self._ylims[0], np.min(y_coords)), max(self._ylims[1], np.max(y_coords)))

        # Add newest step
        step = (
            Step(y_data, theta_data, self._domain, self._grid)
            if self.params.regularization is None
            else RegularizedStep(y_data, theta_data, self._domain, self._grid, self._refined_grid)
        )
        self.steps.append(step)

        # Update time
        self._current_time += self._tau

    def _run_single_step(self):
        self._update_mechanical_step()
        self._mechanical_step.solve()
        self._update_thermal_step()
        self._thermal_step.solve()
        self._append_step(self._thermal_step.y(), self._thermal_step.theta())

    def run(self, num_steps: int = 1) -> None:
        """Run one or more steps of the staggered scheme. In each step first a mechanical and then a
        thermal step is performed."""
        if num_steps <= 0:
            raise ValueError("Invalid number of steps.")

        if num_steps == 1:
            self._run_single_step()
            return

        for _ in tqdm(range(num_steps)):
            self._run_single_step()
