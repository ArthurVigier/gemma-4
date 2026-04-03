"""Simulator-facing interfaces for Gazebo/PX4 and AirSim.

Gazebo Harmonic + PX4 SITL is the primary simulator stack.
AirSim is kept as a secondary backend for compatibility experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .arc_types import DroneAction


@dataclass(slots=True)
class SimObservation:
    """Minimal observation envelope shared by simulator adapters."""

    rgb_image: np.ndarray
    timestamp_ns: int
    simulator_name: str


class DroneSimulator(Protocol):
    """Protocol implemented by AirSim and Gazebo adapters."""

    def get_observation(self) -> SimObservation:
        """Returns the latest RGB observation."""

    def send_action(self, action: DroneAction) -> None:
        """Applies the predicted action to the simulated drone."""
