"""Shared dataclasses for ARC-drone components."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ArcGrid:
    """Discrete ARC-like grid with values in [0, 9]."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("ArcGrid values must be a 2D array.")
        if not np.issubdtype(self.values.dtype, np.integer):
            raise TypeError("ArcGrid values must use an integer dtype.")
        if int(self.values.min()) < 0 or int(self.values.max()) > 9:
            raise ValueError("ArcGrid values must stay in the [0, 9] range.")


@dataclass(slots=True)
class DroneAction:
    """Simple action envelope for simulator commands."""

    velocity_xyz: tuple[float, float, float]
    yaw_rate: float
    halt_probability: float


@dataclass(slots=True)
class TargetZone:
    """Target zone expressed in PX4 NED coordinates."""

    center_ned: tuple[float, float, float]
    radius_m: float


@dataclass(slots=True)
class BenchmarkTask:
    """A synthetic ARC-drone task."""

    task_id: str
    family: str
    input_grid: ArcGrid
    target_grid: ArcGrid
    target_action: DroneAction
    target_zone: TargetZone | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkMetrics:
    """Metrics reported by ARC-Drone-Bench."""

    grid_accuracy: float
    action_accuracy: float
    latency_ms: float
    energy_joules: float
    within_budget: bool
