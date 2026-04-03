"""Metrics helpers for ARC-Drone-Bench."""

from __future__ import annotations

import numpy as np

from .arc_types import ArcGrid, BenchmarkMetrics, DroneAction
from .config import BenchmarkConfig


def grid_accuracy(prediction: ArcGrid, target: ArcGrid) -> float:
    """Returns exact-cell accuracy for ARC-style grids."""

    if prediction.values.shape != target.values.shape:
        raise ValueError("Grid shapes must match for accuracy computation.")
    return float(np.mean(prediction.values == target.values))


def action_accuracy(prediction: DroneAction, target: DroneAction, atol: float = 0.15) -> float:
    """Binary action correctness under small actuation tolerances."""

    pred = np.array([*prediction.velocity_xyz, prediction.yaw_rate], dtype=float)
    tgt = np.array([*target.velocity_xyz, target.yaw_rate], dtype=float)
    return float(np.allclose(pred, tgt, atol=atol))


def package_metrics(
    prediction: ArcGrid,
    target_grid: ArcGrid,
    predicted_action: DroneAction,
    target_action: DroneAction,
    latency_ms: float,
    energy_joules: float,
    config: BenchmarkConfig,
) -> BenchmarkMetrics:
    """Packages the main benchmark metrics and budget status."""

    within_budget = latency_ms <= config.latency_budget_ms and energy_joules <= config.energy_budget_joules
    return BenchmarkMetrics(
        grid_accuracy=grid_accuracy(prediction, target_grid),
        action_accuracy=action_accuracy(predicted_action, target_action),
        latency_ms=latency_ms,
        energy_joules=energy_joules,
        within_budget=within_budget,
    )
